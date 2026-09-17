"""看碰撞形狀：把外觀的 mesh 關掉，只留 group 3 的碰撞體（半透明紅色）。

跑法（macOS 一定要 mjpython）：
    uv run mjpython my_collision_view.py
"""
import time

import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('pineapple_v0/my_robot.xml')
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # geomgroup[i] = 1 顯示、0 隱藏。這個模型：
    #   group 0 = 沒指定 class 的（地板、膝蓋那兩個圓柱）
    #   group 1 = 外觀 mesh（contype=0，不參與碰撞）
    #   group 3 = class="collision" 的碰撞體
    viewer.opt.geomgroup[1] = 0   # 關掉好看的外殼
    viewer.opt.geomgroup[3] = 1   # 打開碰撞體

    # 順便把接觸點和接觸力畫出來
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    viewer.sync()

    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
