## You could add some configs to perform other training experiments...

LeNet_cfg = {
    
    'model_type': 'ResNext50', #myLeNet, myResnet, ResNext50
    'data_root' : '/home/fas3/example-img/SiW/protocol3/train',
    'test_root' : '/home/fas3/example-img/SiW/protocol3/test',
    'img_root'  : '/home/fas3/fas3-CNN/plot_img_SiW_protocol3' ,
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    'best_model': 'best_model_ACER_SiW1_protocol3.pt',
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.001,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 30,
    
}