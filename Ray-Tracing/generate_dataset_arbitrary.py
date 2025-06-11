import os
gpu_num = 0  # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Allows to exit cell execution in Jupyter
class ExitCell(Exception):
    def _render_traceback_(self):
        pass

# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

tf.random.set_seed(1)  # Set global random seed for reproducibility

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.io
from scipy.io import savemat

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement

# Load and configure scene
scene = load_scene("floor_wall.xml")  # Try also sionna.rt.scene.etoile
# print(scene.objects)
scene.get("back").radio_material = "itu_wood"  # "wall" is made of "itu_brick"
scene.get("desk").radio_material = "itu_plywood"
scene.get("floor").radio_material = "itu_concrete"
scene.get("front").radio_material = "itu_wood"
scene.get("wall_001").radio_material = "itu_concrete"
scene.get("windows").radio_material = "itu_glass"

# Configure simulation parameters
samples = 10
count = 1
subcarrier_spacing = 312.5e3
fft_size = 52
num_anttena_large = 10
num_anttena_small = 4
scene.bandwidth = 20e6  # in Hz
scene.synthetic_array = True  # If set to False, ray tracing will be done per antenna element (slower for large arrays)
frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing)
scene.frequency = 5.71e9  # in Hz; implicitly updates RadioMaterials

# Set up save path
if scene.synthetic_array == True:
    isSyn = 'Syn'
else:
    isSyn = 'noSyn'
save_directory = f'/home/LC/Sionna/BFI-MUSIC/dataset/training_dataset_arb4x2_uni4x2_uni10xx2_{isSyn}_metal.mat'

# Initialize data collection arrays
h_matrices_high = []
h_matrices_low_uni = []
h_matrices_low_arb = []
aod_array = []

# Generate dataset
while count <= samples:
    # Set coordinates
    tx_x_cordinate = 2.98
    tx_y_cordinate = 3
    tx_z_cordinate = 0
    rx_x_cordinate = np.random.uniform(-3, 3)
    rx_y_cordinate = np.random.uniform(0, 3.4)
    rx_z_cordinate = np.random.uniform(-7, 7)
    rx_random_oriantation1 = np.random.uniform(-3.141592653589793, 3.141592653589793)
    rx_random_oriantation2 = np.random.uniform(-3.141592653589793, 3.141592653589793)
    rx_random_oriantation3 = np.random.uniform(-3.141592653589793, 3.141592653589793)

    # Configure high resolution array
    scene.tx_array = PlanarArray(
        num_rows=num_anttena_large,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="V")

    # Configure receiver array
    scene.rx_array = PlanarArray(
        num_rows=2,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="V")

    # Create transmitter
    tx = Transmitter(
        name="tx",
        position=[tx_x_cordinate, tx_y_cordinate, tx_z_cordinate],
        orientation=[0, 0, 0],
        power_dbm=20
    )

    # Add transmitter instance to scene
    scene.add(tx)

    # Create a receiver
    rx = Receiver(
        name="rx",
        position=[rx_x_cordinate, rx_y_cordinate, rx_z_cordinate],
        orientation=[rx_random_oriantation1, rx_random_oriantation2, rx_random_oriantation3]
    )

    # Add receiver instance to scene
    scene.add(rx)
    # tx.look_at(rx)
    # rx.look_at(tx)  # Transmitter points towards receiver

    # Compute propagation paths for high resolution array
    paths_high = scene.compute_paths(
        max_depth=4,
        # method="exhaustive",
        num_samples=1e7,
        diffraction=True,
        scattering=True
    )  
    
    # Check if line of sight exists
    if paths_high is not None and paths_high.types.numpy().size > 0 and paths_high.types.numpy()[0, 0] == 0:
        haslos = True
    else:
        haslos = False

    if haslos == True:
        # Configure uniform small array
        scene.tx_array = PlanarArray(
            num_rows=num_anttena_small,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="V")

        # Configure receiver array
        scene.rx_array = PlanarArray(
            num_rows=2,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="V")

        # Create transmitter
        tx = Transmitter(
            name="tx",
            position=[tx_x_cordinate, tx_y_cordinate, tx_z_cordinate],
            orientation=[0, 0, 0],
            power_dbm=20
        )

        # Create a receiver
        rx = Receiver(
            name="rx",
            position=[rx_x_cordinate, rx_y_cordinate, rx_z_cordinate],
            orientation=[rx_random_oriantation1, rx_random_oriantation2, rx_random_oriantation3]
        )
        
        # Compute propagation paths for uniform small array
        paths_low_uni = scene.compute_paths(
            max_depth=4,
            # method="exhaustive",
            num_samples=1e7,
            diffraction=True,
            scattering=True
        )
        
        # Configure arbitrary small array
        scene.tx_array = PlanarArray(
            num_rows=2,
            num_cols=2,
            vertical_spacing=1,
            horizontal_spacing=2,
            pattern="dipole",
            polarization="V")

        # Configure receiver array
        scene.rx_array = PlanarArray(
            num_rows=2,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="V")
        
        # Create transmitter
        tx = Transmitter(
            name="tx",
            position=[tx_x_cordinate, tx_y_cordinate, tx_z_cordinate],
            orientation=[0, 0, 0],
            power_dbm=20
        )

        # Create a receiver
        rx = Receiver(
            name="rx",
            position=[rx_x_cordinate, rx_y_cordinate, rx_z_cordinate],
            orientation=[rx_random_oriantation1, rx_random_oriantation2, rx_random_oriantation3]
        )
        
        # Compute propagation paths for arbitrary small array
        paths_low_arb = scene.compute_paths(
            max_depth=4,
            # method="exhaustive",
            num_samples=1e7,
            diffraction=True,
            scattering=True
        )
        
        # Clean up scene
        scene.remove("tx")
        scene.remove("rx")
        
        # Progress indicator
        if count % 5 == 0:
            print(f"{count} samples have been generated")
        count += 1
        
    else:
        # Clean up scene if no line of sight
        scene.remove("tx")
        scene.remove("rx")
        continue

    # Extract channel impulse responses
    a_high, tau_high = paths_high.cir()
    a_low_uni, tau_low_uni = paths_low_uni.cir()
    a_low_arb, tau_low_arb = paths_low_arb.cir()

    # Convert to frequency domain
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
    
    # Process and store data
    h_freq_sqz_high = np.squeeze(h_freq_high)
    h_matrices_high.append(h_freq_sqz_high)
    h_freq_sqz_low_uni = np.squeeze(h_freq_low_uni)
    h_matrices_low_uni.append(h_freq_sqz_low_uni)
    h_freq_sqz_low_arb = np.squeeze(h_freq_low_arb)
    h_matrices_low_arb.append(h_freq_sqz_low_arb)

    # Calculate angle of departure
    aod = np.degrees(np.arctan((tx_z_cordinate-rx_z_cordinate)/abs(tx_x_cordinate-rx_x_cordinate)))
    aod_array.append(aod)

# Prepare dataset for saving    
dataset = {
    'aods': aod_array,
    'h': [
        {'h_high': h_matrices_high[i], 'h_low_uni': h_matrices_low_uni[i], 'h_low_arb': h_matrices_low_arb[i]} 
        for i in range(len(aod_array))
    ]  
}

# Save dataset    
savemat(save_directory, dataset)
print(f"Dataset has been saved to {save_directory}")



