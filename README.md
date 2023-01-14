# UNT_DNNBenchmarks
 benchmarks for research projects at UNT

# Contributors:
Dr. Hui Zhao
Anna Nordstrand
Justin Garrigus
Khoa Ho

###

=================
===== 12/26 =====
=================

This file adds Alexnet, VGG16, and YOLOv5l to run_klh.py. In essence, this 
project replaces PyTorch's Conv2D layer with a custom-implemented version that 
runs a matrix-multiplication subroutine (gemm.cu) that can be shown on the 
simulator.


=== Changelog === 

Changes are detailed in reference to "MLNoC Teams/shared files/run_klh.py":
* Reorganized general project structure
  * Separated main() from Resnet
  * Moved Resnet implementation to a class instead  
* Created interface/class design
  * All architecture classes (resnet18, vgg16, alexnet, yolo) contain: 
    * __init__: Sets up the class, assigning self.model to the official 
      pretrained PyTorch model. 
    * __call__(dataset): Runs an inference on a torch.Tensor dataset.
    * preprocess(image): Preprocesses a single image, turning it into a 
      torch.Tensor in the process. 
    * __repr__: Displays each layer in the architecture.
* Implemented Alexnet, VGG16, and YOLOv5l by invoking their layers manually, 
  replacing calls to PyTorch.Conv2D to the custom conv2d(x, filePath, module) 
  function instead.
* Architecture is chosen from command-line arguments. 
* Displayed inference results as a formatted list.


=== Steps to Replicate === 

1.) Clone Anna's files from "hengshan.cse.unt.edu:/home/anna/resnet18". 
2.) Update Anna's "run.py" file to the new version.
3.) Install requirements: 
    $ pip install git+https://github.com/nottombrown/imagenet_stubs
    $ pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
4.) cd into the project directory and obtain YOLOv5l: 
    $ wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l-cls.pt -P models/
5.) Rebuild the project:
    $ rm -rf build
    $ mkdir build 
    $ cd build 
    $ cmake ..
    $ make 
    $ cd ..
6.) Run the project, passing a specific model as an argument (e.g., "yolov5l", 
    "alexnet", "vgg16", or "resnet18"). For example: 
    $ python3 run.py -yolov5l


=== Errors === 

* If make gives an error about cutlass not being found, append to the end of 
  CMakeLists.txt within the main project directory: 
    target_include_directories(${PROJECT_NAME} PRIVATE "path/to/your/cutlass/project/directory") 
* If the first GEMM operation yields an error (returncode=1) about an unknown 
  option "-gpgpu_shmem_option", ensure that local GPGPU_Sim configuration files 
  are valid for your version of GPGPU_Sim. I needed to replace the local
  "gpgpusim.config" file with a different version for my project to work, 
  obtained from running: 
    $ wget https://raw.githubusercontent.com/gpgpu-sim/gpgpu-sim_distribution/master/configs/tested-cfgs/SM7_TITANV/gpgpusim.config -O gpgpusim.config


=== Notes === 

* The accompanying diagrams (Alexnet.png, VGG16.png, and YOLOv5l.png) each 
  describe the layout of the architecture as well as the names of the resulting 
  convolutional files. Use it as a reference alongside the source code to see 
  what the program is doing and what is being generated.