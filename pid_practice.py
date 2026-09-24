"""課堂練習：鎖住兩個輪子，腿的四個關節用 PID 控制。

跑法：  uv run mjpython pid_practice.py
沒視窗：uv run python pid_practice.py --headless   （只印數字，驗證用）

改最上面的 KP / KI / KD 重跑，看 terminal 印的誤差怎麼變。
"""
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

NUM_MOTOR = 6
# 馬達順序（看 bipedwheel.xml 的 <actuator>）：
#   0 L_thigh  1 L_calf  2 L_wheel  3 R_thigh  4 R_calf  5 R_wheel
LEGS = [0, 1, 3, 4]
WHEELS = [2, 5]

# ---- 腿：兩個姿勢輪流切換，每 SWITCH_SEC 秒換一次（看 step response 用） ----
POSE_A = np.array([1.27, -2.127, 0, 1.27, -2.127, 0])   # test.py 原本的姿勢
POSE_B = np.array([0.60, -1.200, 0, 0.60, -1.200, 0])   # 腿伸直一點
SWITCH_SEC = 3.0

# ---- 增益：練習就是改這裡 ----
KP = np.array([10.0, 10.0, 0.0, 10.0, 10.0, 0.0])
KI = np.array([0.0,  0.0,  0.0, 0.0,  0.0,  0.0])      # 先 0，觀察穩態誤差再加
KD = np.array([0.1,  0.1,  0.0, 0.1,  0.1,  0.0])

# ---- 輪子：鎖在角度 0（硬彈簧＋阻尼），不參與 PID 練習 ----
WHEEL_KP = 20.0
WHEEL_KD = 0.5

I_LIMIT = 2.0        # 積分項最多貢獻幾 N·m（anti-windup，防止 I 越積越大）
SIM_DT = 0.005       # 200 Hz，跟 test.py 一樣
PRINT_SEC = 0.5


class PID:
    def __init__(self, kp, ki, kd, dt, i_limit):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.i_limit = i_limit
        self.integral = np.zeros_like(kp)

    def __call__(self, target_q, q, dq):
        err = target_q - q
        self.integral += err * self.dt
        # anti-windup：把「積分項的力矩」夾在 ±i_limit 以內
        i_term = np.clip(self.ki * self.integral, -self.i_limit, self.i_limit)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.integral = np.where(self.ki > 0, i_term / self.ki, 0.0)
        # D 項：目標速度是 0，所以就是 -dq（跟 test.py 的 target_dq=0 一樣）
        return self.kp * err + i_term + self.kd * (0.0 - dq)


def main():
    headless = "--headless" in sys.argv
    model = mujoco.MjModel.from_xml_path("robot/pineapple_v0/scene.xml")
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    pid = PID(KP, KI, KD, SIM_DT, I_LIMIT)
    ctrl_lo, ctrl_hi = model.actuator_ctrlrange.T

    def step():
        t = data.time
        target = POSE_A if int(t // SWITCH_SEC) % 2 == 0 else POSE_B
        q = data.sensordata[:NUM_MOTOR]
        dq = data.sensordata[NUM_MOTOR:2 * NUM_MOTOR]

        tau = np.zeros(NUM_MOTOR)
        tau[LEGS] = pid(target, q, dq)[LEGS]
        tau[WHEELS] = WHEEL_KP * (0.0 - q[WHEELS]) - WHEEL_KD * dq[WHEELS]
        data.ctrl[:] = np.clip(tau, ctrl_lo, ctrl_hi)
        mujoco.mj_step(model, data)

        if int(t / SIM_DT) % int(PRINT_SEC / SIM_DT) == 0:
            err = target - q
            print(f"t={t:5.2f}  腿誤差(rad) Lth={err[0]:+.3f} Lca={err[1]:+.3f} "
                  f"Rth={err[3]:+.3f} Rca={err[4]:+.3f}  |  "
                  f"輪子角度 L={q[2]:+.3f} R={q[5]:+.3f}")

    if headless:
        while data.time < 2 * SWITCH_SEC:
            step()
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            step()
            viewer.sync()
            left = SIM_DT - (time.time() - start)
            if left > 0:
                time.sleep(left)


if __name__ == "__main__":
    main()
