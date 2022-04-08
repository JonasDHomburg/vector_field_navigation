// ============================================================================
// Name        : image_viewer_gui.cpp
// Author      : Daniel Rudolph <drudolph@techfak.uni-bielefeld.de>
// Description : Creates a GUI to view two different images.
// ============================================================================

// ROS
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
// ROS - OpenCV_ Bridge
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Boost Mutex
#include <boost/thread/mutex.hpp>

#include "image_viewer_gui.hpp"

using namespace std;

static image_transport::Subscriber sub1, sub2;

static cv::Mat image1, image2;
boost::mutex mtx_image1, mtx_image2;

static bool imageIsLive1, imageIsLive2;
static bool image1once, image2once;
static int image1_rgb2bgr, image2_rgb2bgr;

static string defaultSubTopic1;
static string defaultSubTopic2;

void imageCallback1(const sensor_msgs::ImageConstPtr &msg) {
  if (imageIsLive1 || image1once) {
//    ROS_INFO("[%s] imageCallback1.", ros::this_node::getName().c_str());
    mtx_image1.lock();
    image1 = cv_bridge::toCvShare(msg, "bgr8")->image;
    mtx_image1.unlock();
    if (image1once)
      image1once = false;
  }
}

void imageCallback2(const sensor_msgs::ImageConstPtr &msg) {
  if (imageIsLive2 || image2once) {
//    ROS_INFO("[%s]  imageCallback2.", ros::this_node::getName().c_str());
    mtx_image2.lock();
    image2 = cv_bridge::toCvShare(msg, "bgr8")->image;
    mtx_image2.unlock();
    if (image2once)
      image2once = false;
  }
}

ImageViewGUI::ImageViewGUI(QWidget *parent) : QWidget(parent), shutdown_required(false), thread(&ImageViewGUI::spin, this) {

  // 1st Row
  groupBox1 = new QGroupBox(this);
  groupBox1->setTitle("Image 1 Parameter");

  scopeLabel1 = new QLabel(this);
  scopeLabel1->setText("Topic:");

  textBox1 = new QLineEdit(this);
  textBox1->setText(QString::fromStdString(defaultSubTopic1));
  textBox1->setReadOnly(true);

  loadImageButton1 = new QPushButton(this);
  loadImageButton1->setText("loadImageButton1");
  loadImageButton1->setDisabled(true);
  QObject::connect(loadImageButton1, &QPushButton::clicked, this, &ImageViewGUI::clickedLoadImageButton1);

  radioButton1 = new QRadioButton(this);
  radioButton1->setText("live");
  radioButton1->setChecked(true);
  radioButton1->setDisabled(true);
  QObject::connect(radioButton1, &QRadioButton::clicked, this, &ImageViewGUI::clickedRadioButton1);
  imageIsLive1 = radioButton1->isChecked();

  transparencySliderLabel1 = new QLabel(this);
  transparencySliderLabel1->setText("Transparency 0% to 100%");

  transparencySlider1 = new QSlider(Qt::Horizontal, this);
  transparencySlider1->setMinimum(0);
  transparencySlider1->setMaximum(100);
  transparencySlider1->setSliderPosition(50);
  QObject::connect(transparencySlider1, &QSlider::sliderMoved, this, &ImageViewGUI::transparencySliderValueChanged1);

  transparencySliderLabelInfo1 = new QLabel(this);
  transparencySliderLabelInfo1->setText(QString::number(transparencySlider1->value()));

  qhBox1 = new QHBoxLayout;
  qhBox1->addWidget(scopeLabel1);
  qhBox1->addWidget(textBox1);
  qhBox1->addWidget(loadImageButton1);
  qhBox1->addWidget(radioButton1);
  qhBox1->addWidget(transparencySliderLabel1);
  qhBox1->addWidget(transparencySlider1);
  qhBox1->addWidget(transparencySliderLabelInfo1);
  groupBox1->setLayout(qhBox1);

  // 2nd Row
  groupBox2 = new QGroupBox(this);
  groupBox2->setTitle("Image 2 Parameter");

  scopeLabel2 = new QLabel(this);
  scopeLabel2->setText("Topic:");

  textBox2 = new QLineEdit(this);
  textBox2->setText(QString::fromStdString(defaultSubTopic2));
  textBox2->setReadOnly(true);

  loadImageButton2 = new QPushButton(this);
  loadImageButton2->setText("loadImageButton2");
  loadImageButton2->setDisabled(true);
  QObject::connect(loadImageButton2, &QPushButton::clicked, this, &ImageViewGUI::clickedLoadImageButton2);

  radioButton2 = new QRadioButton(this);
  radioButton2->setText("live");
  radioButton2->setChecked(true);
  radioButton2->setDisabled(true);
  QObject::connect(radioButton2, &QRadioButton::clicked, this, &ImageViewGUI::clickedRadioButton2);
  imageIsLive2 = radioButton2->isChecked();

  transparencySliderLabel2 = new QLabel(this);
  transparencySliderLabel2->setText("Transparency 0% to 100%");

  transparencySlider2 = new QSlider(Qt::Horizontal, this);
  transparencySlider2->setMinimum(0);
  transparencySlider2->setMaximum(100);
  transparencySlider2->setSliderPosition(50);
  QObject::connect(transparencySlider2, &QSlider::sliderMoved, this, &ImageViewGUI::transparencySliderValueChanged2);

  transparencySliderLabelInfo2 = new QLabel(this);
  transparencySliderLabelInfo2->setText(QString::number(transparencySlider2->value()));

  qhBox2 = new QHBoxLayout;
  qhBox2->addWidget(scopeLabel2);
  qhBox2->addWidget(textBox2);
  qhBox2->addWidget(loadImageButton2);
  qhBox2->addWidget(radioButton2);
  qhBox2->addWidget(transparencySliderLabel2);
  qhBox2->addWidget(transparencySlider2);
  qhBox2->addWidget(transparencySliderLabelInfo2);
  groupBox2->setLayout(qhBox2);

  // merge row1 and row2
  qvBox1 = new QVBoxLayout;
  qvBox1->addWidget(groupBox1);
  qvBox1->addWidget(groupBox2);
  groupBox0 = new QGroupBox(this);
  groupBox0->setTitle("Parameter");
  groupBox0->setLayout(qvBox1);
  groupBox0->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);

//   Image Labels
  imageLabel1 = new QLabel(this);

  // Total Gui
  this->setWindowTitle(QString::fromStdString(ros::this_node::getName()));
  gridLayout1 = new QGridLayout();
  gridLayout1->addWidget(groupBox0, 0, 0);
  gridLayout1->addWidget(imageLabel1, 1, 0);
  this->setLayout(gridLayout1);

//  image1once = true;
//  while (image1once)
//    usleep(100000);
//
//  this->updateImage1();
//
//  image2once = true;
//  while (image2once)
//    usleep(100000);
//
//  this->updateImage2();
}

ImageViewGUI::~ImageViewGUI() {
  delete textBox1;
  delete textBox2;
  delete scopeLabel1;
  delete scopeLabel2;
  delete loadImageButton1;
  delete loadImageButton2;
  delete radioButton1;
  delete radioButton2;
  delete transparencySlider1;
  delete transparencySlider2;
  delete transparencySliderLabel1;
  delete transparencySliderLabel2;
  delete transparencySliderLabelInfo1;
  delete transparencySliderLabelInfo2;
  delete groupBox0;
  delete groupBox1;
  delete groupBox2;
  delete gridLayout1;
  delete qvBox1;
  delete qhBox1;
  delete qhBox2;
  delete imageLabel1;

  shutdown_required = true;
  thread.join();
}

void ImageViewGUI::spin() {
  ros::Rate loop(10);
  while (ros::ok()) {
    ros::spinOnce();
    updateImage1();
    updateImage2();
    loop.sleep();
  }
  ROS_INFO("[%s] Shutdown this node.", ros::this_node::getName().c_str());
  ros::shutdown();
  QApplication::quit();
}

void ImageViewGUI::updateImage1() {
  mtx_image1.lock();
  mtx_image1copy.lock();
  image1.copyTo(this->image1copy);
  if (image1copy.type() == CV_8UC1) {
    cvtColor(this->image1copy, this->image1copy, CV_GRAY2BGR);
  } else if (image1_rgb2bgr) {
    cv::cvtColor(this->image1copy, this->image1copy, cv::COLOR_BGR2RGB);
  }
  mtx_image1.unlock();
  mtx_image1copy.unlock();
  updateImageLabel();
}

void ImageViewGUI::updateImage2() {
  mtx_image2.lock();
  mtx_image2copy.lock();
  image2.copyTo(this->image2copy);
  if (image2copy.type() == CV_8UC1) {
    cvtColor(this->image2copy, this->image2copy, CV_GRAY2BGR);
  } else if (image2_rgb2bgr) {
    cv::cvtColor(this->image2copy, this->image2copy, cv::COLOR_BGR2RGB);
  }
  mtx_image2.unlock();
  mtx_image2copy.unlock();
  updateImageLabel();
}

void ImageViewGUI::updateImageLabel() {
  // Display the image
  if (!image1copy.empty() && !image2copy.empty()) {
//    if (image1.size != image2.size)
//      cv::resize(image2, image2, image1.size());
    double sum = transparencySlider1->value() + transparencySlider2->value();
    mtx_image1copy.lock();
    mtx_image2copy.lock();
    if(image1copy.size() != image2copy.size())
      cv::resize(image2copy, image2copy, image1copy.size());
    cv::addWeighted(image1copy, transparencySlider1->value() / sum, image2copy, transparencySlider2->value() / sum, 0.0, blended);
    mtx_image1copy.unlock();
    mtx_image2copy.unlock();
    QImage qtImage1 = mat2QImage(blended);
    if(qtImage1.size().width() > 1200) {
      imageLabel1->setPixmap(QPixmap::fromImage(qtImage1.scaledToWidth(1200)));
    } else if(qtImage1.size().height() > 1000) {
      imageLabel1->setPixmap(QPixmap::fromImage(qtImage1.scaledToHeight(1000)));
    } else {
      imageLabel1->setPixmap(QPixmap::fromImage(qtImage1));
    }
    imageLabel1->show();
  } else {
//    ROS_INFO("[%s] Cannot update imagelabel. Images are empty image1 %d image2 %d", ros::this_node::getName().c_str(), image1copy.empty(), image2copy.empty());
  }
}

void ImageViewGUI::clickedLoadImageButton1() {
  image1once = true;
  while (image1once)
    usleep(100000);

//  this->updateImage1();
//  ROS_INFO("[%s] clicked loadImageButton1", ros::this_node::getName().c_str());
}

void ImageViewGUI::clickedLoadImageButton2() {
  image2once = true;
  while (image2once)
    usleep(100000);

//  this->updateImage2();
//  ROS_INFO("[%s] clicked loadImageButton2", ros::this_node::getName().c_str());
}

void ImageViewGUI::clickedRadioButton1() {
  imageIsLive1 = radioButton1->isChecked();
  this->updateImage1();
}

void ImageViewGUI::clickedRadioButton2() {
  imageIsLive2 = radioButton2->isChecked();
  this->updateImage2();
}

void ImageViewGUI::transparencySliderValueChanged1(int value) {
  this->transparencySliderLabelInfo1->setText(QString::number(value));
  this->transparencySlider2->setValue(100 - value);
  this->transparencySliderLabelInfo2->setText(QString::number(100 - value));
  updateImageLabel();
}

void ImageViewGUI::transparencySliderValueChanged2(int value) {
  this->transparencySliderLabelInfo2->setText(QString::number(value));
  this->transparencySlider1->setValue(100 - value);
  this->transparencySliderLabelInfo1->setText(QString::number(100 - value));
  updateImageLabel();
}


int main(int argc, char *argv[]) {
  ROS_INFO_STREAM("Start: " << ros::this_node::getName());
  // Init ROS
  ros::init(argc, argv, ros::this_node::getName());
  ros::NodeHandle node("~");

  image_transport::ImageTransport it(node);

  node.param<string>("subcriber_topic_1", defaultSubTopic1, "/image");
  node.param<string>("subcriber_topic_2", defaultSubTopic2, "/genicam_cam1/cam");
  node.param<int>("image1_rgb2bgr", image1_rgb2bgr, 1);
  node.param<int>("image2_rgb2bgr", image2_rgb2bgr, 1);

  ROS_INFO("[%s] subcriber_topic_1 %s", ros::this_node::getName().c_str(), defaultSubTopic1.c_str());
  ROS_INFO("[%s] subcriber_topic_2 %s", ros::this_node::getName().c_str(), defaultSubTopic2.c_str());

  sub1 = it.subscribe(defaultSubTopic1, 1, imageCallback1);
  sub2 = it.subscribe(defaultSubTopic2, 1, imageCallback2);

  QApplication app(argc, argv);
  ImageViewGUI gui;
  gui.show();
  app.connect(&app, SIGNAL(lastWindowClosed()), &app, SLOT(quit()));

  return app.exec();
}
