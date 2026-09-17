"""會開、會轉彎、跌倒會自己爬起來 —— 用鍵盤操控。

===================================================================
鍵盤
===================================================================
    W / S      前進 / 後退（每按一次 ±0.2 m/s）
    A / D      左轉 / 右轉（每按一次 ±0.5 rad/s）
    空白鍵     全部歸零（煞車，原地站著）
    P          往前踹它 700 N，測試爬起來用

===================================================================
跟 my_getup.py 差在哪：三件事
===================================================================
1. **傾角改用「投影重力」算，轉彎才不會壞。**
   舊寫法是看機身的上方向在世界座標 x 分量（atan2(R[0,2], R[2,2])）——
   那等於假設機器人永遠面向 +x。實測：真正前傾 10° 的時候，
   轉 90° 會讀成 0°、轉 180° 會讀成 -10°（正負顛倒，控制器會加速倒下）。
   新寫法把重力向量轉進機身座標再算角度，轉到哪個方向讀值都一樣。

2. **速度迴圈從「PD 位置」改成「PI 速度」。**
   舊的外層是拿世界座標 x 當位置目標 —— 一轉彎那個 x 就沒有意義了。
   新的改成對「速度誤差」積分，不看世界座標，轉到哪都成立。
   附帶好處：直線極速從 1.0 m/s 提高到 1.6 m/s。

3. **多了轉彎控制（yaw）。** 兩個輪子給不一樣的力矩就會轉：
       左輪 = 平衡力矩 − 轉彎力矩
       右輪 = 平衡力矩 + 轉彎力矩
   轉彎力矩同樣是 PI：看 gyro 的 z 軸讀值和命令差多少。

===================================================================
實測的能力範圍（headless 掃出來的）
===================================================================
    前進     最快 1.6 m/s（1.8 就倒）
    後退     最快 1.0 m/s（1.3 就倒）
    原地轉   到 8 rad/s 都還追得準，沒測到上限
    邊走邊轉 1.3 m/s + 1.0 rad/s 可以；1.3 + 2.0 會倒
    跌倒爬起 -180°~180° 每 10 度丟一次，37/37 都站得回來

跑法：
    uv run mjpython my_walk.py
"""
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = 'pineapple_v0/my_robot.xml'
SIMULATION_DT = 0.005

# ---- 平衡內層：傾角 → 輪子力矩 ----
KP_PITCH, KD_PITCH = 120.0, 12.0
# ---- 速度外層（PI）：速度誤差 → 傾角目標 ----
KP_VEL, KI_VEL = 0.8, 0.6
PITCH_REF_MAX, VEL_INT_MAX = 0.25, 0.4
# ---- 轉彎（PI）：角速度誤差 → 左右輪的力矩差 ----
KP_YAW, KI_YAW, YAW_INT_MAX = 8.0, 5.0, 2.0
# ---- 腿：站著的時候維持蹲姿 ----
KP_LEG, KD_LEG = 30.0, 0.6
THIGH_TARGET, CALF_TARGET = 1.22, -1.92

# ---- 操控 ----
V_STEP, W_STEP = 0.2, 0.5
V_LIMIT, W_LIMIT = (-1.0, 1.5), 3.0
PUSH_FORCE, PUSH_SECONDS = 700.0, 0.15

# ---- 卡住偵測與起身動作（跟 my_getup.py 同一組，已驗證 37/37）----
STUCK_TILT_DEG, STUCK_GYRO, STUCK_VEL, STUCK_HOLD = 40.0, 0.5, 0.15, 0.5
GETUP_SECONDS = 0.4
GETUP_THIGH, GETUP_CALF = 0.5, -0.2
GETUP_KP, GETUP_KD, GETUP_WHEEL_TORQUE = 120.0, 3.0, -4.0


def read_state(data):
    """把姿態換成控制要用的四個量，全部跟「機器人面向哪邊」無關。

    做法是把重力向量 (0,0,-1) 從世界座標轉進機身座標。機身站直時重力在
    機身看來就是正下方；往前傾 10 度，重力在機身看來就往後偏 10 度。
    用這個偏移量算角度，機器人轉到哪個方位讀值都一樣。
    """
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, data.sensordata[18:22])
    rot = rot.reshape(3, 3)
    gravity_in_body = rot.T @ np.array([0.0, 0.0, -1.0])

    pitch = math.atan2(gravity_in_body[0], -gravity_in_body[2])   # 前後傾，正 = 前傾
    tilt = math.acos(float(np.clip(rot[2, 2], -1.0, 1.0)))        # 離垂直多遠，永遠是正的
    heading = math.atan2(rot[1, 0], rot[0, 0])                    # 面向哪個方位
    forward = np.array([math.cos(heading), math.sin(heading)])
    forward_vel = float(np.dot(data.sensordata[31:33], forward))  # 沿著「自己的前方」跑多快
    return pitch, tilt, forward_vel


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIMULATION_DT
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    cmd = {'v': 0.0, 'w': 0.0}
    push = {'left': 0.0}

    def key_callback(keycode):
        ch = chr(keycode) if 0 <= keycode < 0x110000 else ''
        if ch == 'W':
            cmd['v'] = min(cmd['v'] + V_STEP, V_LIMIT[1])
        elif ch == 'S':
            cmd['v'] = max(cmd['v'] - V_STEP, V_LIMIT[0])
        elif ch == 'A':
            cmd['w'] = min(cmd['w'] + W_STEP, W_LIMIT)
        elif ch == 'D':
            cmd['w'] = max(cmd['w'] - W_STEP, -W_LIMIT)
        elif ch == ' ':
            cmd['v'] = cmd['w'] = 0.0
        elif ch == 'P':
            push['left'] = PUSH_SECONDS
            print(f'  往前踹 {PUSH_FORCE:.0f} N')
            return
        else:
            return
        print(f"  前進 {cmd['v']:+.1f} m/s   轉彎 {cmd['w']:+.1f} rad/s")

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    vel_int = 0.0
    yaw_int = 0.0
    stuck_timer = 0.0
    getup_timer = None
    attempts = 0

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            joint_pos = data.sensordata[:6]
            joint_vel = data.sensordata[6:12]
            pitch, tilt, forward_vel = read_state(data)
            pitch_rate = data.sensordata[23]
            yaw_rate = data.sensordata[24]

            # ---- 卡住偵測：躺著、而且已經不動了 ----
            lying = math.degrees(tilt) > STUCK_TILT_DEG
            still = abs(pitch_rate) < STUCK_GYRO and abs(forward_vel) < STUCK_VEL
            if getup_timer is None:
                stuck_timer = stuck_timer + SIMULATION_DT if (lying and still) else 0.0
                if stuck_timer > STUCK_HOLD:
                    getup_timer, attempts = 0.0, attempts + 1
                    print(f'  卡住了（傾角 {math.degrees(tilt):.0f}°）→ 第 {attempts} 次起身')
            else:
                getup_timer += SIMULATION_DT
                if getup_timer >= GETUP_SECONDS:
                    getup_timer, stuck_timer = None, 0.0
                    vel_int = yaw_int = 0.0        # 積分器歸零，不然躺著那段會累積成大偏移

            # ---- 平衡 ＋ 速度（PI）----
            vel_err = cmd['v'] - forward_vel
            vel_int = float(np.clip(vel_int + vel_err * SIMULATION_DT, -VEL_INT_MAX, VEL_INT_MAX))
            pitch_ref = float(np.clip(KP_VEL * vel_err + KI_VEL * vel_int,
                                      -PITCH_REF_MAX, PITCH_REF_MAX))
            balance_torque = KP_PITCH * (pitch - pitch_ref) + KD_PITCH * pitch_rate

            # ---- 轉彎（PI）：變成左右輪的力矩差 ----
            yaw_err = cmd['w'] - yaw_rate
            yaw_int = float(np.clip(yaw_int + yaw_err * SIMULATION_DT, -YAW_INT_MAX, YAW_INT_MAX))
            turn_torque = KP_YAW * yaw_err + KI_YAW * yaw_int

            left_wheel = balance_torque - turn_torque
            right_wheel = balance_torque + turn_torque
            thigh_ref, calf_ref, kp, kd = THIGH_TARGET, CALF_TARGET, KP_LEG, KD_LEG

            # ---- 起身動作進行中就蓋掉腿姿和輪子 ----
            if getup_timer is not None:
                thigh_ref, calf_ref = GETUP_THIGH, GETUP_CALF
                kp, kd = GETUP_KP, GETUP_KD
                left_wheel = right_wheel = GETUP_WHEEL_TORQUE
                vel_int = yaw_int = 0.0

            data.ctrl[:] = [
                kp * (thigh_ref - joint_pos[0]) - kd * joint_vel[0],
                kp * (calf_ref - joint_pos[1]) - kd * joint_vel[1],
                left_wheel,
                kp * (thigh_ref - joint_pos[3]) - kd * joint_vel[3],
                kp * (calf_ref - joint_pos[4]) - kd * joint_vel[4],
                right_wheel,
            ]

            data.xfrc_applied[:] = 0
            mujoco.mjv_applyPerturbForce(model, data, viewer.perturb)
            if push['left'] > 0.0:
                data.xfrc_applied[base_id, 0] = PUSH_FORCE
                push['left'] -= SIMULATION_DT

            mujoco.mj_step(model, data)
            viewer.sync()

            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == '__main__':
    main()
