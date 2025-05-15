import os
import time
import numpy as np
import scipy.io
from scipy.io import savemat, loadmat
import tensorflow as tf
import matplotlib.pyplot as plt

# GPU 配置
gpu_num = ""  # 使用 "" 表示使用 CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Jupyter 单元格退出
class ExitCell(Exception):
    def _render_traceback_(self):
        pass

# 导入 Sionna
try:
    import sionna
except ImportError as e:
    # 如果 Sionna 包未安装则安装
    import os
    os.system("pip install sionna")
    import sionna

# 配置使用单个 GPU 并按需分配内存
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# 避免 TensorFlow 警告
tf.get_logger().setLevel('ERROR')

# 设置全局随机种子以确保可重现性
tf.random.set_seed(1)

# 导入 Sionna RT 组件
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

# 链路级仿真导入
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement

# 载入并配置场景
scene = load_scene("floor_wall.xml")  # 也可以尝试 sionna.rt.scene.etoile
scene.get("back").radio_material = "itu_wood"
scene.get("desk").radio_material = "itu_plywood"
scene.get("floor").radio_material = "itu_concrete"
scene.get("front").radio_material = "itu_wood"
scene.get("wall_001").radio_material = "itu_concrete"
scene.get("windows").radio_material = "itu_glass"

# 配置仿真参数
samples = 1000  # 总样本数
count = 1      # 计数器
subcarrier_spacing = 312.5e3
fft_size = 52
num_anttena_large = 10
num_anttena_small = 4
scene.bandwidth = 20e6  # 带宽 Hz
scene.synthetic_array = True  # 如果设为 False，会对每个天线元素进行射线追踪（较慢）
frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing)
scene.frequency = 5.71e9  # 频率 Hz，隐式更新 RadioMaterials

# 设置保存路径
if scene.synthetic_array == True:
    isSyn = 'Syn'
else:
    isSyn = 'noSyn'
base_directory = '/home/LC/Sionna/BFI-MUSIC/dataset'
save_filename = f'training_dataset_arb6x6_uni4x2_uni10x2_{isSyn}.mat'
save_directory = os.path.join(base_directory, save_filename)
temp_save_directory = os.path.join(base_directory, f'temp_{save_filename}')

# 检查是否存在临时文件以恢复进度
h_matrices_high = []
h_matrices_low_uni = []
h_matrices_low_arb = []
aod_array = []
start_count = 1

if os.path.exists(temp_save_directory):
    try:
        print(f"找到临时文件，尝试恢复进度...")
        temp_data = loadmat(temp_save_directory)
        
        if 'aods' in temp_data and len(temp_data['aods']) > 0:
            aod_array = temp_data['aods'].tolist()[0]
            start_count = len(aod_array) + 1
            
            # 提取已有的通道数据
            h_data = temp_data['h'][0]
            for i in range(len(aod_array)):
                h_matrices_high.append(h_data[i]['h_high'][0])
                h_matrices_low_uni.append(h_data[i]['h_low_uni'][0])
                h_matrices_low_arb.append(h_data[i]['h_low_arb'][0])
                
            print(f"成功恢复进度！将从样本 {start_count} 开始继续...")
        else:
            print("临时文件格式不正确，将从头开始...")
    except Exception as e:
        print(f"恢复进度时出错: {e}")
        print("将从头开始...")

count = start_count

# 实时保存函数
def save_temp_progress():
    try:
        # 准备保存的数据集
        dataset = {
            'aods': aod_array,
            'h': [
                {'h_high': h_matrices_high[i], 'h_low_uni': h_matrices_low_uni[i], 'h_low_arb': h_matrices_low_arb[i]} 
                for i in range(len(aod_array))
            ]  
        }
        
        # 保存临时数据
        savemat(temp_save_directory, dataset)
        print(f"临时进度已保存到 {temp_save_directory}")
    except Exception as e:
        print(f"保存临时进度时出错: {e}")

# 自动保存间隔（样本数）
auto_save_interval = 20
last_save_time = time.time()
time_start = time.time()

# 生成数据集
try:
    while count <= samples:
        # 设置坐标
        tx_x_cordinate = 2.98
        tx_y_cordinate = 3
        tx_z_cordinate = 0
        rx_x_cordinate = np.random.uniform(-3, 3)
        rx_y_cordinate = np.random.uniform(0, 3.4)
        rx_z_cordinate = np.random.uniform(-7, 7)
        rx_random_oriantation1 = np.random.uniform(-3.141592653589793, 3.141592653589793)
        rx_random_oriantation2 = np.random.uniform(-3.141592653589793, 3.141592653589793)
        rx_random_oriantation3 = np.random.uniform(-3.141592653589793, 3.141592653589793)

        # 配置高分辨率阵列
        scene.tx_array = PlanarArray(
            num_rows=num_anttena_large,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="V")

        # 配置接收器阵列
        scene.rx_array = PlanarArray(
            num_rows=2,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="V")

        # 创建发射器
        tx = Transmitter(
            name="tx",
            position=[tx_x_cordinate, tx_y_cordinate, tx_z_cordinate],
            orientation=[0, 0, 0],
            power_dbm=20
        )

        # 将发射器实例添加到场景
        scene.add(tx)

        # 创建接收器
        rx = Receiver(
            name="rx",
            position=[rx_x_cordinate, rx_y_cordinate, rx_z_cordinate],
            orientation=[rx_random_oriantation1, rx_random_oriantation2, rx_random_oriantation3]
        )

        # 将接收器实例添加到场景
        scene.add(rx)
        # tx.look_at(rx)
        # rx.look_at(tx)  # 发射器朝向接收器

        # 为高分辨率阵列计算传播路径
        paths_high = scene.compute_paths(
            max_depth=3,
            # method="exhaustive",
            num_samples=1e6,
            diffraction=True,
            scattering=True
        )  
        
        # 检查是否存在视线
        if paths_high is not None and paths_high.types.numpy().size > 0 and paths_high.types.numpy()[0, 0] == 0:
            haslos = True
        else:
            haslos = False

        if haslos == True:
            # 配置均匀小阵列
            scene.tx_array = PlanarArray(
                num_rows=num_anttena_small,
                num_cols=1,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="dipole",
                polarization="V")

            # 配置接收器阵列
            scene.rx_array = PlanarArray(
                num_rows=2,
                num_cols=1,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="dipole",
                polarization="V")

            # 创建发射器
            tx = Transmitter(
                name="tx",
                position=[tx_x_cordinate, tx_y_cordinate, tx_z_cordinate],
                orientation=[0, 0, 0],
                power_dbm=20
            )

            # 创建接收器
            rx = Receiver(
                name="rx",
                position=[rx_x_cordinate, rx_y_cordinate, rx_z_cordinate],
                orientation=[rx_random_oriantation1, rx_random_oriantation2, rx_random_oriantation3]
            )
            
            # 为均匀小阵列计算传播路径
            paths_low_uni = scene.compute_paths(
                max_depth=3,
                # method="exhaustive",
                num_samples=1e6,
                diffraction=True,
                scattering=True
            )
            
            # 配置小异形阵列
            scene.tx_array = PlanarArray(
                num_rows=6,
                num_cols=6,
                vertical_spacing=1,
                horizontal_spacing=1,
                pattern="dipole",
                polarization="V")

            # 配置接收器阵列
            scene.rx_array = PlanarArray(
                num_rows=2,
                num_cols=1,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="dipole",
                polarization="V")
            
            # 创建发射器
            tx = Transmitter(
                name="tx",
                position=[tx_x_cordinate, tx_y_cordinate, tx_z_cordinate],
                orientation=[0, 0, 0],
                power_dbm=20
            )

            # 创建接收器
            rx = Receiver(
                name="rx",
                position=[rx_x_cordinate, rx_y_cordinate, rx_z_cordinate],
                orientation=[rx_random_oriantation1, rx_random_oriantation2, rx_random_oriantation3]
            )
            
            # 计算任意小阵列的传播路径
            paths_low_arb = scene.compute_paths(
                max_depth=3,
                # method="exhaustive",
                num_samples=1e7,
                diffraction=True,
                scattering=True
            )
            
            # 清理场景
            scene.remove("tx")
            scene.remove("rx")
            
            # 提取通道脉冲响应
            a_high, tau_high = paths_high.cir()
            a_low_uni, tau_low_uni = paths_low_uni.cir()
            a_low_arb, tau_low_arb = paths_low_arb.cir()

            # 转换为频域
            h_freq_high = cir_to_ofdm_channel(
                frequencies,
                a_high,
                tau_high,
                normalize=True
            )
            h_freq_low_uni = cir_to_ofdm_channel(
                frequencies,
                a_low_uni,
                tau_low_uni,
                normalize=True
            )
            h_freq_low_arb = cir_to_ofdm_channel(
                frequencies,
                a_low_arb,
                tau_low_arb,
                normalize=True
            )
            
            # 处理并存储数据
            h_freq_sqz_high = np.squeeze(h_freq_high)
            h_matrices_high.append(h_freq_sqz_high)
            h_freq_sqz_low_uni = np.squeeze(h_freq_low_uni)
            h_matrices_low_uni.append(h_freq_sqz_low_uni)
            h_freq_sqz_low_arb = np.squeeze(h_freq_low_arb)
            h_matrices_low_arb.append(h_freq_sqz_low_arb)

            # 计算出发角
            aod = np.degrees(np.arctan((tx_z_cordinate-rx_z_cordinate)/abs(tx_x_cordinate-rx_x_cordinate)))
            aod_array.append(aod)
            
            # 进度指示器与时间估计
            current_time = time.time()
            elapsed_time = current_time - time_start
            avg_time_per_sample = elapsed_time / (count - start_count + 1) if count > start_count else 0
            estimated_time_remaining = avg_time_per_sample * (samples - count)
            
            if count % 5 == 0:
                print(f"样本 {count}/{samples} 已生成 ({(count/samples*100):.1f}%)")
                print(f"估计剩余时间: {estimated_time_remaining/60:.1f} 分钟")
            
            # 定期自动保存
            if count % auto_save_interval == 0 or (current_time - last_save_time) > 300:  # 每auto_save_interval个样本或每5分钟保存一次
                save_temp_progress()
                last_save_time = current_time
                
            count += 1
            
        else:
            # 如果没有视线，清理场景
            scene.remove("tx")
            scene.remove("rx")
            continue

    # 准备最终数据集保存    
    dataset = {
        'aods': aod_array,
        'h': [
            {'h_high': h_matrices_high[i], 'h_low_uni': h_matrices_low_uni[i], 'h_low_arb': h_matrices_low_arb[i]} 
            for i in range(len(aod_array))
        ]  
    }

    # 保存最终数据集    
    savemat(save_directory, dataset)
    print(f"完整数据集已保存到 {save_directory}")
    
    # 如果成功完成，删除临时文件
    if os.path.exists(temp_save_directory):
        os.remove(temp_save_directory)
        print("临时文件已删除")

except KeyboardInterrupt:
    print("\n检测到程序中断！正在保存当前进度...")
    save_temp_progress()
    print(f"已处理 {count-1}/{samples} 个样本 ({(count-1)/samples*100:.1f}%)")
    print("您可以稍后通过运行相同的脚本从断点处继续")
    
except Exception as e:
    print(f"\n发生错误: {e}")
    print("正在保存当前进度...")
    save_temp_progress()
    print(f"已处理 {count-1}/{samples} 个样本 ({(count-1)/samples*100:.1f}%)")