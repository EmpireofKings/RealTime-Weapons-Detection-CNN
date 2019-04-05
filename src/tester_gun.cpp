#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image.h>

#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;
using namespace dlib;


// This is a 6x6 convolutional layer that does 2x downsampling.
template <long num_filters, typename SUBNET> using con6d = con<num_filters,9,9,2,2,SUBNET>;

// This is a 3x3 convolutional layer that does not downsampling.
template <long num_filters, typename SUBNET> using con3  = con<num_filters,3,3,1,1,SUBNET>;

// This is a con6d 8x downsampling block as defined by con6d. ReLu is the prefered method
// as it only has to calculate on the positive returns of the function. 

template <typename SUBNET> using downsampler  = relu<bn_con<con6d<32, relu<bn_con<con6d<32, relu<bn_con<con6d<32,SUBNET>>>>>>>>>;

// A 3x3 ReLu convolutional layer.
template <typename SUBNET> using rcon3  = relu<bn_con<con3<32,SUBNET>>>;

using net_type = loss_mmod<con<1,9,9,1,1,rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<5>>>>>>>>;

// ----------------------------------------------------------------------------------------

VideoCapture *cap = new VideoCapture();

int main(int argc, char** argv) try
{
    cout << "Starting" << endl;

    if (argc != 3)
    {
        
        cout << "program.  For example, if you are in the examples folder then execute " << endl;
        cout << "this program by running: " << endl;
        cout << "   ./dnn_mmod_ex faces" << endl;
        cout << endl;
        return 0;
    }
    //const std::string faces_directory = argv[1];
    const std::string fileName = argv[1];
    const std::string netFile = argv[2];
   
    //std::vector<matrix<rgb_pixel>> images_test;

    //std::vector<std::vector<mmod_rect>> face_boxes_test;
     
    //load_image_dataset(images_test, face_boxes_test, faces_directory+"/testing.xml");



    net_type net;



    deserialize(netFile.c_str()) >> net;



    //cout << "testing results:  " << test_object_detection_function(net, images_test, face_boxes_test) << endl;
 
    image_window win;

    //cout << "Debug 1" << endl;

    cap->open(fileName);

    //cout << "Debug 2" << endl;

    int counter = 0;

    int max_frames = 4;
 

    while(1)
    {
      Mat frame;
        Mat frame_resize;
      // Capture frame-by-frame
        cap->operator>>(frame);

        cv::resize(frame, frame_resize, cv::Size(320,180));

      if ( counter >= max_frames)
      {

      

        cv_image<bgr_pixel> image(frame_resize);
        
        matrix<rgb_pixel> dlibImage;

        // dlib::array2d<rgb_pixel> dlibImage;
        assign_image(dlibImage, image);

        auto&& img = dlibImage;

        

        pyramid_up(img);

        auto dets = net(img);

        win.clear_overlay();

        win.set_image(img);

        for (auto&& d : dets)
        {
            win.add_overlay(d.rect);
            cout << "Debug "  << d.rect << endl;

        }
            
        

        
        counter = 0;
      }
      else 
      {

          counter++;
      }
    }

/*
    for (auto&& img : images_test)
    {
        pyramid_up(img);
        auto dets = net(img);
        win.clear_overlay();
        win.set_image(img);
        for (auto&& d : dets)
            win.add_overlay(d);
        cin.get();
    }
    */
    return 0;


}
catch(std::exception& e)
{
    cout << e.what() << endl;
}
