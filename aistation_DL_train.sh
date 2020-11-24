#!/bin/bash
# This script is used for submitting the DL training misson in a quick step!

echo  -e  "################################################################ \n
    Please read the following instruction before you training models: \n  
     
      You can train models in following ways: \n
      ./aistation_DL_train.sh tensorflow mnist 1 
      which means training tensorflow model with mnist by using 1 GPU \n

      You can use more GPUS like this:
      ./aistation_DL_train.sh tensorflow mnist 2 \n

      Also you can train the models with MPI by this:
       ./aistation_DL_train.sh tensorflow mnist mpi 2 \n
     
      Please Note: 
      the default user is 'inspur', so please modify the 
      path which is used in training scripts if you want 
      to test those scripts in your own directory.

      Now we support following models: \n
      tensorflow mnist(cifar10)
      pytorch mnist(cifar10) 
      caffe mnist(cifar10)
      mxnet mnist
      paddle mnist \n
      in one or more GPUs, or in MPI (only mnist) training mode. \n
#############################################################"            

#echo "Starting the training"
echo "You are training $1 $2 GPU:$3 "
path=`pwd`

# Tensorflow
############################################################
if [ "$1" = "tensorflow" ]; then
############################################################
echo "$1"
  #------------------------------------------------- #
  if [ "$2" = "mnist" ];  then
   # mnist
   echo "$2"
    # one gpu
    if [ $3 -eq 1 ]; then
     echo "GPU: $3"
     python ./tensorflow/mnist/tf_mnist_single.py 
    # multi gpu 
    elif [ $3 -gt 1 ]; then
     echo "GPU: $3"
     python ./tensorflow/mnist/tf_mnist_multi.py --num_gpus=$3
    elif [ "$3" = "mpi" ]; then                                                        
     echo "MPI: $4"                                                              
     mpirun -np $4 -allow-run-as-root python ./horovod/tensorflow_mnist.py
    else
     echo "GPU: 1"
     python ./tensorflow/mnist/tf_mnist_single.py
    fi
  #------------------------------------------------- #
  elif [ "$2" = "cifar10" ]; then 
   # mnist                                                                                
   echo "$2"                                                                            
    # one gpu                                                                            
    if [ $3 -eq 1 ]; then                                                                
     echo "GPU: $3"                                                                      
     python ./tensorflow/cifar10/TF_SingleGPU/cifar10_single_gpu_train.py                                        
    # multi gpu                                                                          
    elif [ $3 -gt 1 ]; then                                                                
     echo "GPU: $3"                                                                      
     python ./tensorflow/cifar10/TF_MultiGPU/cifar10_multi_gpu_train.py --num_gpus=$3                           
    else                                                                                 
     echo "GPU: 1"                                                                       
     python ./tensorflow/cifar10/TF_SingleGPU/cifar10_single_gpu_train.py                                        
    fi                                                                                   
  fi

# Pytorch
############################################################
elif [ "$1" = "pytorch" ]; then                                                                                            
############################################################
echo "$1"                                                                                                                   
  #------------------------------------------------- #
  if [ "$2" = "mnist" ];  then                                                                                              
   # mnist                                                                                                                  
   echo "$2"                                                                                                                
    # one gpu                                                                                                               
    if [ $3 -eq 1 ]; then                                                                                                   
     echo "GPU: $3"                                                                                                         
     python ./pytorch/mnist/pytorch_mnist_single.py                                                                           
    # multi gpu                                                                                                             
    elif [ $3 -gt 1 ]; then                                                                                                   
     echo "GPU: $3"                                                                                                         
     python ./pytorch/mnist/pytorch_mnist_multi.py --num_gpus=$3  
    # mpi mode                                                            
    elif [ "$3" = "mpi" ]; then                                                                               
     echo "MPI: $4"                                                                                          
     mpirun -np $4 -allow-run-as-root python ./horovod/pytorch_mnist.py                                   
    else                                                                                                                    
     echo "GPU: 1"                                                                                                          
     python ./pytorch/mnist/pytorch_mnist_single.py                                                                           
    fi                                                                                                                      
  #------------------------------------------------- #
  elif [ "$2" = "cifar10" ]; then                                                                                           
   # mnist                                                                                                                  
   echo "$2"                                                                                                                
    # one gpu                                                                                                               
    if [ $3 -eq 1 ]; then                                                                                                   
     echo "GPU: $3"                                                                                                         
     python ./pytorch/cifar10/pytorch_cifar10_single.py                                                   
    # multi gpu                                                                                                             
    elif [ $3 -gt 1 ]; then                                                                                                   
     echo "GPU: $3"                                                                                                         
     python ./pytorch/cifar10/pytorch_cifar10_multi.py --num_gpus=$3                                       
    else                                                                                                                    
     echo "GPU: 1"                                                                                                          
     python ./pytorch/cifar10/pytorch_cifar10_single.py                                                   
    fi                                                                                                                      
  fi                                                                                                                        

# Caffe
############################################################
elif [ "$1" = "caffe" ]; then                                                   
############################################################
echo "$1" 
cd $path
  #------------------------------------------------- #
  if [ "$2" = "mnist" ];  then                                                    
   # mnist                                                                        
   echo "$2"                                                                      
    # one gpu                                                                     
    if [ $3 -eq 1 ]; then                                                        
     echo "GPU: $3"                                                               
     cd ./caffe/mnist && caffe train --solver=./solver_lenet.prototxt && cd $path
    # multi gpu                                                                   
    elif [ $3 -gt 1] ; then                                                         
     echo "GPU: $3"                                                               
     cd ./caffe/mnist && caffe train --solver=./solver_lenet.prototxt -gpu all  && cd $path
    # mpi mode
    elif [ "$3" = "mpi" ]; then                                                    
     echo "MPI: $4"                                                               
     cd ./caffe/mnist && mpirun -np $4 -allow-run-as-root caffe train --solver=./solver_lenet.prototxt -gpu all && cd $path
    else                                                                          
     echo "GPU: 1"                                                                
     cd ./caffe/mnist && caffe train --solver=./solver_lenet.prototxt && cd $path
    fi                                                                            
  #------------------------------------------------- #
  elif [ "$2" = "cifar10" ]; then                                                 
   # mnist                                                                        
   echo "$2"                                                                      
    # one gpu                                                                     
    if [ $3 -eq 1 ]; then                                                         
     echo "GPU: $3"                                                               
     cd ./caffe/cifar10/ && caffe train  --solver=./cifar10_full_sigmoid_solver.prototxt  && cd $path
    # multi gpu                                                                   
    elif [ $3 -gt 1 ]; then                                                         
     echo "GPU: $3"                                                               
     cd ./caffe/cifar10/ && caffe train  --solver=./cifar10_full_sigmoid_solver.prototxt -gpu all  && cd $path
    else                                                                          
     echo "GPU: 1"                                                                
     cd ./caffe/cifar10/ && caffe train  --solver=./cifar10_full_sigmoid_solver.prototxt    && cd $path
    fi                                                                            
  fi                                                                              

# Mxnet                                                             
############################################################          
elif [ "$1" = "mxnet" ]; then                                       
############################################################          
echo "$1"                                                             
  #------------------------------------------------- #                
  if [ "$2" = "mnist" ];  then                                        
   # mnist                                                            
   echo "$2"                                                          
    # one gpu                                                         
    if [ $3 -eq 1 ]; then                                             
     echo "GPU: $3"                                                   
     python ./mxnet/mx_mnist_single.py                   
    # multi gpu                                                       
    elif [ $3 -gt 1 ]; then                                             
     echo "GPU: $3"                                                   
     python ./mxnet/mx_mnist_multi.py --num_gpus=$3      
    # mpi mode                                                        
    elif [ "$3" = "mpi" ]; then                                        
     echo "MPI: $4"                                                   
     mpirun -np $4 -allow-run-as-root python ./horovod/mxnet_mnist.py
    else                                                              
     echo "GPU: 1"                                                    
     python ./mxnet/mx_mnist_single.py                   
    fi                                                                
  fi
                                                                                                      
# Paddlepaddle                                                                                               
############################################################                                          
elif [ "$1" = "paddle" ]; then                                                                         
############################################################                                          
echo "$1"                                                                                             
  #------------------------------------------------- #                                                
  if [ "$2" = "mnist" ];  then                                                                        
   # mnist                                                                                            
   echo "$2"                                                                                          
    # one gpu                                                                                         
    if [ $3 -eq 1 ]; then                                                                             
     echo "GPU: $3"                                                                                   
     python ./paddle/paddle_mnist_single.py                                                                
    # multi gpu                                                                                       
    elif [ $3 -gt 1 ]; then                                                                             
     echo "GPU: $3"                                                                                   
     python ./paddle/paddle_mnist_multi.py 
    else                                                                                              
     echo "GPU: 1"                                                                                    
     python ./paddle/paddle_mnist_single.py                                                                
    fi                                                                                                
  fi                                                                                                  

fi   
