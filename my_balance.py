"""平衡控制器：讓這台輪足機器人站得住、而且能照速度命令前進。

===================================================================
它在做什麼
===================================================================
這台車本質上是「掃把立在手掌上」—— 專有名詞叫 inverted pendulum（倒立擺）。
掃把要不倒，手要往「它正在倒的那個方向」移動，把支點重新塞回重心底下。
這裡的「手」就是兩個輪子。

控制分成兩層，這種疊法叫 cascade control（串級控制）：

  外層（慢，管「要去哪」）
     看目前速度和位置 → 決定「應該傾斜幾度」
     ↓ 把傾角目標 pitch_ref 交給內層
  內層（快，管「不要倒」）
     看目前傾角和傾角變化率 → 決定「輪子要出多少力矩」

最反直覺的一點：**要往前走，輪子得先往後轉一下。**
輪子往後 → 身體往前傾 → 傾了之後往前倒的分量才推著整台車前進。
外層就是在做這件事：想往前，就故意給一個「往前傾」的目標。

===================================================================
訊號從哪裡來（索引對照 my_robot.xml 的 <sensor> 順序）
===================================================================
  sensordata[18:22]  imu_quat       機身姿態 → 算出傾角 pitch
  sensordata[23]     imu_gyro 的 y  傾角的變化率（倒得多快）
  sensordata[31]     frame_lin_vel 的 x  前進速度
  qpos[0]                           前進位置（模擬裡直接讀，真機要靠里程計）

===================================================================
增益是怎麼來的
===================================================================
不是猜的，是掃過參數網格挑出來的（每組都跑 15–20 秒模擬，比存活時間、
最大傾角、漂移距離）。手調的順序是：
  1. 先只開 KP_PITCH，確認正負號對（號錯會加速倒下，撐不到 0.5 秒）
  2. 加 KD_PITCH 壓抖動
  3. 這時候站得住但會一路滑走 → 加外層 KP_VEL 把速度拉回 0
  4. 還是會停在離原點一公尺外 → 加 KP_POS 把位置也拉回來

跑法：
    uv run mjpython my_balance.py
在視窗裡按**空白鍵**往前推、按 **B** 往後推（各 700 N、0.15 秒）。
滑鼠拖曳也可以：先在機身上**雙擊**選中它，再 Ctrl＋拖曳。
"""
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = 'pineapple_v0/my_robot.xml'
SIMULATION_DT = 0.005

# ---- 內層：傾角 → 輪子力矩 ----
KP_PITCH = 120.0   # 傾 1 弧度就出 120 N·m 的意圖（實際會被馬達上限 3.69 夾住）
KD_PITCH = 12.0    # 阻尼，壓住來回震盪

# ---- 外層：速度／位置 → 傾角目標 ----
KP_VEL = 0.8       # 速度差 1 m/s → 目標傾角偏 0.8 弧度（會被下面的上限夾住）
KP_POS = 0.4       # 位置差 1 m → 目標傾角偏 0.4 弧度
PITCH_REF_MAX = 0.25   # 目標傾角上限 ±0.25 弧度（約 14 度），不准它整台趴下去

# ---- 腿：維持蹲姿，不參與平衡 ----
KP_LEG, KD_LEG = 30.0, 0.6
THIGH_TARGET, CALF_TARGET = 1.22, -1.92

# ---- 速度命令（公尺/秒）。0 = 原地站著 ----
V_CMD = 0.0

# ---- 互動用的推力 ----
PUSH_FORCE = 700.0        # 牛頓。整台車重約 63 N，所以這是重重一撞（兩個方向都推得倒）
PUSH_SECONDS = 0.15


def body_pitch(quat):
    """從機身姿態四元數算出前後傾角，正值 = 往 +x 前傾。

    做法：把機身自己的「上」方向轉換成世界座標，得到一個向量。
    機身站直時它是 (0, 0, 1)；往前傾時 x 分量會變正。
    用 atan2 把這兩個分量換回角度就是傾角。
    """
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    return math.atan2(rot[0, 2], rot[2, 2])


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIMULATION_DT
    mujoco.mj_resetDataKeyframe(model, data, 0)

    x_ref = float(data.qpos[0])   # 位置目標，有速度命令時會跟著往前推

    # 空白鍵往前推、B 往後推。滑鼠拖曳也可以（要先「雙擊」選中機身）
    push = {'left': 0.0, 'dir': 0.0}

    def key_callback(keycode):
        ch = chr(keycode) if 0 <= keycode < 0x110000 else ''
        if ch == ' ':
            push['left'], push['dir'] = PUSH_SECONDS, 1.0
            print(f'  往前推 {PUSH_FORCE:.0f} N')
        elif ch == 'B':
            push['left'], push['dir'] = PUSH_SECONDS, -1.0
            print(f'  往後推 {PUSH_FORCE:.0f} N')

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            joint_pos = data.sensordata[:6]
            joint_vel = data.sensordata[6:12]
            pitch = body_pitch(data.sensordata[18:22])
            pitch_rate = data.sensordata[23]
            forward_vel = data.sensordata[31]
            forward_pos = data.qpos[0]

            # ---- 外層：要走多快、該在哪 → 該傾幾度 ----
            x_ref += V_CMD * SIMULATION_DT
            pitch_ref = np.clip(
                KP_VEL * (V_CMD - forward_vel) + KP_POS * (x_ref - forward_pos),
                -PITCH_REF_MAX, PITCH_REF_MAX)

            # ---- 內層：該傾幾度 vs 實際傾幾度 → 輪子力矩 ----
            wheel_torque = KP_PITCH * (pitch - pitch_ref) + KD_PITCH * pitch_rate

            # ---- 腿只管維持蹲姿 ----
            l_thigh = KP_LEG * (THIGH_TARGET - joint_pos[0]) - KD_LEG * joint_vel[0]
            l_calf = KP_LEG * (CALF_TARGET - joint_pos[1]) - KD_LEG * joint_vel[1]
            r_thigh = KP_LEG * (THIGH_TARGET - joint_pos[3]) - KD_LEG * joint_vel[3]
            r_calf = KP_LEG * (CALF_TARGET - joint_pos[4]) - KD_LEG * joint_vel[4]

            # ctrl 的順序照 <actuator> 寫的：L_thigh L_calf L_wheel R_thigh R_calf R_wheel
            data.ctrl[:] = [l_thigh, l_calf, wheel_torque,
                            r_thigh, r_calf, wheel_torque]

            # 外力：每一步先清零，再把滑鼠拖曳和鍵盤推力加回去。
            # mjv_applyPerturbForce 這一行就是「讓 Ctrl＋拖曳真的推得動」的關鍵 ——
            # 被動模式的 viewer 只幫忙算，不幫忙套用。
            data.xfrc_applied[:] = 0
            mujoco.mjv_applyPerturbForce(model, data, viewer.perturb)
            if push['left'] > 0.0:
                data.xfrc_applied[base_id, 0] = PUSH_FORCE * push['dir']
                push['left'] -= SIMULATION_DT

            mujoco.mj_step(model, data)
            viewer.sync()

            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == '__main__':
    main()
