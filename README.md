CalibSetGenerator
Generate your own calib_set for AI vision projects — perfect for models like LPRNet or any AI that needs consistently sized cropped images for training or calibration.

✨ Features

Auto Mode: Automatically detects objects (e.g., license plates) using heuristics.
GUI Mode: Manually select or confirm bounding boxes with mouse input.
Fixed Mode: Applies fixed coordinates for cropping across all images.
Output: RGB images resized to 300×75 px (ideal for LPRNet).
Logging: Saves a CSV log with coordinates and selection mode (auto / gui / fixed).


📦 Installation

Ensure you have Python 3.8+ installed.
Install required dependencies:pip install opencv-python numpy


Download or copy the script build_lpr_calib.py into your project directory.


🚀 Usage
Auto Mode
Automatically detects and crops objects (e.g., license plates) using heuristics.
python build_lpr_calib.py --src "/path/to/images" --dst "/path/to/output" --mode auto --limit 250 --shuffle

Arguments:

--limit: Maximum number of images to process.
--shuffle: Randomly shuffles input images.

GUI Mode
Manually select or confirm bounding boxes using mouse input.
python build_lpr_calib.py --src "/path/to/images" --dst "/path/to/output" --mode gui

Controls:

Enter: Confirm the suggested bounding box.
Space: Draw a custom bounding box.
N: Skip the current image.
Q or ESC: Quit the application.

Fixed Mode
Applies fixed coordinates for cropping across all images.
python build_lpr_calib.py --src "/path/to/images" --dst "/path/to/output" --mode fixed --coords 420 520 300 900

Arguments:

--coords y1 y2 x1 x2: Fixed coordinates for cropping.


📂 Output Structure
After processing, the output directory will contain:
output_dir/
├── plate_000000.jpg
├── plate_000001.jpg
├── ...
└── log.csv  # Log file with coordinates and selection mode


🛠 Example Workflow
Preparing a calib_set for LPRNet on Hailo:

Generate the calib_set using CalibSetGenerator.
Copy the output directory to the model's calib_set folder.
Run quantization or training.


📜 License
This project is licensed under the MIT License. Feel free to use, modify, and share.
