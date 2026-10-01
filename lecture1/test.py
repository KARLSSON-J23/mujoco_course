import mujoco          # 物理模擬器本體
import mujoco.viewer   # 開視窗看模擬畫面用的
import time            # 用來控制速度，讓模擬跟真實時間一樣快
import numpy as np     # 處理「一排數字」（陣列）的工具

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    # PD 控制：力矩 = (目標角度 - 現在角度) × kp  +  (目標速度 - 現在速度) × kd
    #   前半（P）：離目標越遠，推越用力，像橡皮筋
    #   後半（D）：轉越快，越往反方向拉，像煞車，防止衝過頭
    # 每個參數都是 6 個數字，一個馬達一格，六個馬達一次算完
    return (target_q - q) * kp + (target_dq - dq) * kd

NUM_MOTOR = 6   # 馬達數量。順序照 bipedwheel.xml 的 <actuator>：
                #   0 左大腿  1 左小腿  2 左輪  3 右大腿  4 右小腿  5 右輪
# Load a sample model 
# 載入模型。my_scene.xml 的 freejoint 是打開的 → 機器人站在地上，不是吊在空中
model = mujoco.MjModel.from_xml_path('../robot/pineapple_v0/my_scene.xml')
data = mujoco.MjData(model)   # model 是「機器人長怎樣」，data 是「現在的狀態」（位置、速度…）
mujoco.mj_resetDataKeyframe(model, data, 0)   # 從站好的姿勢開始，不要從空中掉下來
# 每個馬達的目標角度（rad）。輪子那兩格（第 2、5 格）下面會被覆蓋掉，這裡的 0 沒用到
target_dof_pos = np.array([1.22, -1.92, 0, 1.22, -1.92, 0])

simulation_dt = 0.005   # 每一步走 0.005 秒 = 一秒算 200 次
# 腿的 PD 增益（一個馬達一格）。輪子那兩格一樣會被下面覆蓋掉
kps = np.array([30, 30 ,10, 30, 30, 10])        # 橡皮筋多硬。原本 10 太軟，落地會垮
kds = np.array([0.6, 0.6, 0.1, 0.6, 0.6, 0.1])  # 煞車多強。原本 0.1 太弱，腿會亂踢
# 輪子的平衡增益 —— 可以玩的就是這三個數字
kp_pitch = 100   # 機身傾 1 rad，輪子出多少力矩
kd_pitch = 10    # 倒得越快，輪子越用力
kv = 20          # 車跑越快，輪子越往回拉（不然會越滑越快然後倒）
# Run a simple simulation
# 開視窗，視窗開著就一直跑（關掉視窗程式才會結束）
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()   # 記下這一步開始的時間
    
        # 1. 腿：用 PD 維持蹲姿
        #    sensordata[0:6] 是六個關節的角度，sensordata[6:12] 是六個關節的速度
        #    目標速度給 0 = 希望它停在目標角度上不要動
        tau = pd_control(target_dof_pos, data.sensordata[:NUM_MOTOR], kps, np.zeros(6), data.sensordata[NUM_MOTOR:NUM_MOTOR + NUM_MOTOR], kds)
        # 2. 輪子：不鎖角度，改成管「不要倒」（倒立擺）
        #    概念：機身往哪邊倒，輪子就往那邊追，把輪子塞回重心底下
        pitch = 2 * data.sensordata[20]        # 機身前後傾角（imu_quat 的 y 分量，小角度近似）
        pitch_rate = data.sensordata[23]       # 傾倒的速度（imu_gyro 的 y）
        velocity = data.sensordata[31]         # 機身往前的速度（frame_lin_vel 的 x）
        # tau[[2, 5]] = 只改第 2、5 格（左輪、右輪），兩個輪子出一樣的力矩
        tau[[2, 5]] = kp_pitch * pitch + kd_pitch * pitch_rate + kv * velocity   # 輪子改成管平衡
        data.ctrl[:] = tau                     # 把 6 個力矩送給馬達（超過馬達上限的部分模擬器會自己截掉）
        model.opt.timestep = simulation_dt
        mujoco.mj_step(model, data)            # 物理世界往前走一步（0.005 秒）
        viewer.sync()                          # 更新視窗畫面

        # 算太快的話等一下，讓模擬速度跟真實時間一樣
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

