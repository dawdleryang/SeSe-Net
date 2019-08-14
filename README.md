#SeSe_Net: semi-supervised segmentation by deep learning 

#by Zeng Zeng, Yang Xulei, Yu Qiyun, Yao Meng, Zhang Le, 

#Pattern Recognition Letter, Aug. 2019


To repeat the results in the paper: 

  1) Download one of the three datasets used in the paper, e.g., carvana dataset 
  
  2) Create the following folder structures
  
     ../input
  
     ../input/car_data/
     
     ../input/car_data/train_images
     
     ../input/car_data/train_masks
     
     ../input/car_data/train_list.csv
     
  3) Run sse_train_step1.py (labelled samples) 
  
      3-1) train a unet model to generate various masks
      
      3-2) train a resnet model by using the generated masks
      
  4) Run sse_train_step2.py (un-labelled samples)
  
      4-1) resnet predict un-labelled samples and generate loss values to refine unet model 
      
   
