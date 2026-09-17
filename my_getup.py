"""平衡 ＋ 跌倒自己爬起來。

===================================================================
它比 my_balance.py 多了什麼
===================================================================
my_balance.py 被推倒之後就躺在地上不動了。這一支多一個「卡住偵測」和一個
0.4 秒的起身動作。

設計上最重要的一件事：**平衡控制器從頭到尾都不關掉。**

第一版不是這樣寫的 —— 第一版偵測到傾角超過門檻就切進「起身模式」、
把輪子力矩接管過去。結果比原本還糟：19 個測試姿態只過 9 個，
連原本平衡器自己救得回來的角度都倒了。原因是機器人往下倒的那一兩秒，
平衡器其實正在拼命拉回來，一關掉就前功盡棄。

所以最後的結構是：平衡器永遠在跑，起身動作只是「卡住太久時插播一下」。

===================================================================
什麼叫「卡住」
===================================================================
不能只看傾角 —— 倒下去的過程中傾角一定很大，但那時候不該插手。
要同時滿足三件事，連續 0.5 秒：

    傾角 > 40°          躺著
    角速度 < 0.5 rad/s  不在翻滾
    前進速度 < 0.15 m/s 不在移動

也就是「躺著而且**已經不動了**」。這個條件很重要，因為起身動作是
開迴路的固定 0.4 秒 —— 只有從靜止出發，結果才可預測。

===================================================================
起身動作本身
===================================================================
就兩件事，同時做，做 0.4 秒：

    1. 腿從蹲姿伸到接近打直（thigh 0.5, calf -0.2），用大的 kp 硬撐
       → 把機身從地板上頂起來，同時把輪子的接觸點往外推
    2. 兩個輪子一起給 -4 N·m（會被馬達上限 3.69 夾住，就是全力）
       → 反作用力矩把機身往回轉

做完就直接還給平衡器。如果還是躺著，偵測器 0.5 秒後會再觸發一次。
**不需要寫重試邏輯，它自己就會重試。**

曾經試過更花俏的版本（把腿前後甩五次製造搖擺再起身），反而比較差：
19 個姿態只過 15 個。簡單的那個過 19 個。

===================================================================
驗證（headless 跑出來的，不是估的）
===================================================================
  初始姿態掃描：-180° 到 180° 每 10 度丟一次，等它躺穩再開控制器
      平地       37/37 站起來，平均 0.9 次嘗試，中位 1.8 秒
      有障礙物   37/37 站起來，平均 1.0 次嘗試，中位 2.0 秒
  站著被推倒：300 / 400 / 600 / 800 N 的衝擊，平衡器單獨都救不回來，
      加上起身動作全部一次就爬起來

跑法：
    uv run mjpython my_getup.py
在視窗裡按**空白鍵**往前推、按 **B** 往後推（各 700 N、0.15 秒），
這個力道一定推得倒，然後看它自己爬起來。
滑鼠拖曳也可以：先在機身上**雙擊**選中它，再 Ctrl＋拖曳。
"""
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = 'pineapple_v0/my_robot.xml'
SIMULATION_DT = 0.005

# ---- 平衡：內層（傾角 → 輪子力矩）----
KP_PITCH, KD_PITCH = 120.0, 12.0
# ---- 平衡：外層（速度／位置 → 傾角目標）----
KP_VEL, KP_POS, PITCH_REF_MAX = 0.8, 0.4, 0.25
# ---- 站著的時候腿維持蹲姿 ----
KP_LEG, KD_LEG = 30.0, 0.6
THIGH_TARGET, CALF_TARGET = 1.22, -1.92
V_CMD = 0.0

# ---- 互動用的推力 ----
PUSH_FORCE = 700.0        # 牛頓。整台車重約 63 N，所以這是重重一撞（兩個方向都推得倒）
PUSH_SECONDS = 0.15

# ---- 卡住偵測 ----
STUCK_TILT_DEG = 40.0     # 傾角超過這個才算躺著
STUCK_GYRO = 0.5          # rad/s，比這個小才算「不在翻滾」
STUCK_VEL = 0.15          # m/s，比這個小才算「不在移動」
STUCK_HOLD = 0.5          # 三個條件要同時連續成立這麼久

# ---- 起身動作 ----
GETUP_SECONDS = 0.4
GETUP_THIGH, GETUP_CALF = 0.5, -0.2   # 接近打直的腿
GETUP_KP, GETUP_KD = 120.0, 3.0       # 撐的力氣要比站著大很多
GETUP_WHEEL_TORQUE = -4.0             # 會被馬達上限 3.69 夾住，等於全力


def body_tilt(quat):
    """回傳 (前後傾角 pitch, 離垂直的總傾角)，單位都是弧度。

    pitch 有正負（正值 = 往 +x 前傾），平衡控制要用。
    總傾角永遠是正的，用 arccos 算，判斷「有沒有躺著」比較不會被正負搞混。
    """
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    pitch = math.atan2(rot[0, 2], rot[2, 2])
    tilt = math.acos(float(np.clip(rot[2, 2], -1.0, 1.0)))
    return pitch, tilt


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIMULATION_DT
    mujoco.mj_resetDataKeyframe(model, data, 0)

    x_ref = float(data.qpos[0])
    stuck_timer = 0.0
    getup_timer = None      # None = 沒在做起身動作
    attempts = 0

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
            pitch, tilt = body_tilt(data.sensordata[18:22])
            pitch_rate = data.sensordata[23]
            forward_vel = data.sensordata[31]

            # ---- 卡住偵測：躺著、而且已經不動了 ----
            lying = math.degrees(tilt) > STUCK_TILT_DEG
            still = abs(pitch_rate) < STUCK_GYRO and abs(forward_vel) < STUCK_VEL
            if getup_timer is None:
                stuck_timer = stuck_timer + SIMULATION_DT if (lying and still) else 0.0
                if stuck_timer > STUCK_HOLD:
                    getup_timer = 0.0
                    attempts += 1
                    print(f'  卡住了（傾角 {math.degrees(tilt):.0f}°）→ 第 {attempts} 次起身')
            else:
                getup_timer += SIMULATION_DT
                if getup_timer >= GETUP_SECONDS:
                    getup_timer = None          # 做完就還給平衡器
                    stuck_timer = 0.0
                    x_ref = float(data.qpos[0])  # 位置目標重設在現在的地方

            # ---- 平衡控制器：不管在不在起身，輪子這條式子都照算 ----
            x_ref += V_CMD * SIMULATION_DT
            pitch_ref = np.clip(
                KP_VEL * (V_CMD - forward_vel) + KP_POS * (x_ref - data.qpos[0]),
                -PITCH_REF_MAX, PITCH_REF_MAX)
            wheel_torque = KP_PITCH * (pitch - pitch_ref) + KD_PITCH * pitch_rate
            thigh_ref, calf_ref = THIGH_TARGET, CALF_TARGET
            kp, kd = KP_LEG, KD_LEG

            # ---- 起身動作進行中就蓋掉上面的腿姿和輪子力矩 ----
            if getup_timer is not None:
                thigh_ref, calf_ref = GETUP_THIGH, GETUP_CALF
                kp, kd = GETUP_KP, GETUP_KD
                wheel_torque = GETUP_WHEEL_TORQUE

            data.ctrl[:] = [
                kp * (thigh_ref - joint_pos[0]) - kd * joint_vel[0],
                kp * (calf_ref - joint_pos[1]) - kd * joint_vel[1],
                wheel_torque,
                kp * (thigh_ref - joint_pos[3]) - kd * joint_vel[3],
                kp * (calf_ref - joint_pos[4]) - kd * joint_vel[4],
                wheel_torque,
            ]

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
