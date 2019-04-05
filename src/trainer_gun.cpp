#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>

using namespace std;
using namespace dlib;



template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,3,3,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<32,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<20>>>>>>>>;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc != 4)
    {
        cout << "Give the path to the examples/faces directory as the argument to this" << endl;
        cout << "program.  For example, if you are in the examples folder then execute " << endl;
        cout << "this program by running: " << endl;
        cout << "   ./dnn_mmod_ex faces_director gpu_stack max_epocs" << endl;
        cout << endl;
        return 0;
    }
    const std::string faces_directory = argv[1];

    const int gpu_stack = atoi(argv[2]);
    
    const int max_epocs = atoi(argv[3]);
  
    std::vector<matrix<rgb_pixel>> images_train;
    
    std::vector<std::vector<mmod_rect>> face_boxes_train;
 
    load_image_dataset(images_train, face_boxes_train, faces_directory+"/training.xml");

    cout << "gpu stack: " << gpu_stack << endl;
    cout << "num training images: " << images_train.size() << endl;



    mmod_options options(face_boxes_train, 64,64);
    // The detector will automatically decide to use multiple sliding windows if needed.
    // For the face data, only one is needed however.
    cout << "num detector windows: "<< options.detector_windows.size() << endl;
    for (auto& w : options.detector_windows)
        cout << "detector window width by height: " << w.width << " x " << w.height << endl;
    cout << "overlap NMS IOU thresh:             " << options.overlaps_nms.get_iou_thresh() << endl;
    cout << "overlap NMS percent covered thresh: " << options.overlaps_nms.get_percent_covered_thresh() << endl;

    // Now we are ready to create our network and trainer.  
    net_type net(options);
    // The MMOD loss requires that the number of filters in the final network layer equal
    // options.detector_windows.size().  So we set that here as well.
    net.subnet().layer_details().set_num_filters(options.detector_windows.size());
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.1);
    trainer.be_verbose();
    trainer.set_synchronization_file("mmod_sync", std::chrono::minutes(1));
    trainer.set_iterations_without_progress_threshold(max_epocs);


    // Now let's train the network.  We are going to use mini-batches of 150
    // images.   The images are random crops from our training set (see
    // random_cropper_ex.cpp for a discussion of the random_cropper). 
    std::vector<matrix<rgb_pixel>> mini_batch_samples;
    std::vector<std::vector<mmod_rect>> mini_batch_labels; 
    
    
    
    random_cropper cropper;
    //cropper.set_chip_dims(200, 200);
    // Usually you want to give the cropper whatever min sizes you passed to the
    // mmod_options constructor, which is what we do here.
   // cropper.set_min_object_size(80,80);
    
    
    dlib::rand rnd;
    // Run the trainer until the learning rate gets small.  This will probably take several
    // hours.
    while(trainer.get_learning_rate() >= 1e-4)
    {
        cropper(gpu_stack, images_train, face_boxes_train, mini_batch_samples, mini_batch_labels);
        // We can also randomly jitter the colors and that often helps a detector
        // generalize better to new images. 
        for (auto&& img : mini_batch_samples)
            disturb_colors(img, rnd);

        trainer.train_one_step(mini_batch_samples, mini_batch_labels);
    }
    // wait for training threads to stop
    trainer.get_net();
    cout << "done training" << endl;

    // Save the network to disk
    net.clean();
    serialize("mmod_gun_network.dat") << net;

}
catch(std::exception& e)
{
    cout << e.what() << endl;
}
