"""真的會動的版本：機器人從半空中落地，六個關節用 PD control 撐住蹲姿。

跟老師的 test.py 差在哪：
  1. 載入 my_scene.xml → my_bipedwheel.xml，那份把 <freejoint/> 打開了，
     機身才會有位置和姿態可以變（原版 base_link 被焊死在世界座標上）
  2. 用 mj_resetDataKeyframe 套用 <keyframe> 的起始姿勢（離地 0.2 公尺的蹲姿）

跑法（macOS 一定要 mjpython）：
    uv run mjpython my_sim.py
"""
import time

import mujoco
import mujoco.viewer
import numpy as np


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """位置誤差乘 kp，速度誤差乘 kd，加起來就是要出的力矩"""
    return (target_q - q) * kp + (target_dq - dq) * kd


NUM_MOTOR = 6
SIMULATION_DT = 0.005

model = mujoco.MjModel.from_xml_path('pineapple_v0/my_robot.xml')
data = mujoco.MjData(model)
model.opt.timestep = SIMULATION_DT

# 套用 keyframe：機身放在離地 0.2 公尺，腿擺成蹲姿
mujoco.mj_resetDataKeyframe(model, data, 0)

target_dof_pos = np.array([1.22, -1.92, 0.0, 1.22, -1.92, 0.0])
kps = np.array([10.0] * NUM_MOTOR)
kds = np.array([0.1] * NUM_MOTOR)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        q = data.sensordata[:NUM_MOTOR]                      # 關節位置
        dq = data.sensordata[NUM_MOTOR:NUM_MOTOR * 2]        # 關節速度
        data.ctrl[:] = pd_control(target_dof_pos, q, kps, np.zeros(NUM_MOTOR), dq, kds)

        mujoco.mj_step(model, data)
        viewer.sync()

        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
