#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class OCR
{
	public:
		float confThreshold;
		float nmsThreshold;
		int inpWidth;
		int inpHeight;
		string modelRecognition;
		Net detector;
		Net recognizer;
        string alphabet;
        OCR(string modelRecognition, string alphabet);
		void decodeBoundingBoxes(const Mat& scores, const Mat& geometry, std::vector<RotatedRect>& detections, std::vector<float>& confidences);
        void fourPointsTransform(const Mat& frame, Point2f vertices[4], Mat& result);
        void decodeText(const Mat& scores, std::string& text);
        void detect_rec(Mat& frame);
};

OCR::OCR(string modelRecognition, string alphabet)
{
	this->confThreshold = 0.5;
	this->nmsThreshold = 0.4;
	this->inpHeight = 320;
	this->inpWidth = 320;
    this->alphabet = alphabet;
	this->detector = readNet("frozen_east_text_detection.pb");
	this->modelRecognition = modelRecognition;
	if (!modelRecognition.empty())
	{
		this->recognizer = readNet(modelRecognition);
	}	
}

void OCR::decodeBoundingBoxes(const Mat& scores, const Mat& geometry, std::vector<RotatedRect>& detections, std::vector<float>& confidences)
{
    detections.clear();
    CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
    CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
    CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);

    const int height = scores.size[2];
    const int width = scores.size[3];
    for (int y = 0; y < height; ++y)
    {
        const float* scoresData = scores.ptr<float>(0, 0, y);
        const float* x0_data = geometry.ptr<float>(0, 0, y);
        const float* x1_data = geometry.ptr<float>(0, 1, y);
        const float* x2_data = geometry.ptr<float>(0, 2, y);
        const float* x3_data = geometry.ptr<float>(0, 3, y);
        const float* anglesData = geometry.ptr<float>(0, 4, y);
        for (int x = 0; x < width; ++x)
        {
            float score = scoresData[x];
            if (score < this->confThreshold)
                continue;

            // Decode a prediction.
            // Multiple by 4 because feature maps are 4 time less than input image.
            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];

            Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
            Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
            RotatedRect r(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            detections.push_back(r);
            confidences.push_back(score);
        }
    }
}

void OCR::fourPointsTransform(const Mat& frame, Point2f vertices[4], Mat& result)
{
    const Size outputSize = Size(100, 32);

    Point2f targetVertices[4] = { Point(0, outputSize.height - 1),
                                  Point(0, 0), Point(outputSize.width - 1, 0),
                                  Point(outputSize.width - 1, outputSize.height - 1),
                                };
    Mat rotationMatrix = getPerspectiveTransform(vertices, targetVertices);

    warpPerspective(frame, result, rotationMatrix, outputSize);
}

void OCR::decodeText(const Mat& scores, std::string& text)
{
    Mat scoresMat = scores.reshape(1, scores.size[0]);

    std::vector<char> elements;
    elements.reserve(scores.size[0]);

    for (int rowIndex = 0; rowIndex < scoresMat.rows; ++rowIndex)
    {
        Point p;
        minMaxLoc(scoresMat.row(rowIndex), 0, 0, 0, &p);
        if (p.x > 0 && static_cast<size_t>(p.x) <= this->alphabet.size())
        {
            elements.push_back(this->alphabet[p.x - 1]);
        }
        else
        {
            elements.push_back('-');
        }
    }

    if (elements.size() > 0 && elements[0] != '-')
        text += elements[0];

    for (size_t elementIndex = 1; elementIndex < elements.size(); ++elementIndex)
    {
        if (elementIndex > 0 && elements[elementIndex] != '-' &&
            elements[elementIndex - 1] != elements[elementIndex])
        {
            text += elements[elementIndex];
        }
    }
}

void OCR::detect_rec(Mat& frame)
{
    std::vector<Mat> outs;
    std::vector<String> outNames(2);
    outNames[0] = "feature_fusion/Conv_7/Sigmoid";
    outNames[1] = "feature_fusion/concat_3";
    Mat blob;
    TickMeter tickMeter;
    blobFromImage(frame, blob, 1.0, Size(this->inpWidth, this->inpHeight), Scalar(123.68, 116.78, 103.94), true, false);
    this->detector.setInput(blob);
    tickMeter.start();
    //this->detector.forward(outs, this->detector.getUnconnectedOutLayersNames());   ////运行会出错
    this->detector.forward(outs, outNames);
    tickMeter.stop();

    Mat scores = outs[0];
    Mat geometry = outs[1];
    // Decode predicted bounding boxes.
    std::vector<RotatedRect> boxes;
    std::vector<float> confidences;
    this->decodeBoundingBoxes(scores, geometry, boxes, confidences);

    // Apply non-maximum suppression procedure.
    std::vector<int> indices;
    NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);

    Point2f ratio((float)frame.cols / this->inpWidth, (float)frame.rows / this->inpHeight);
    // Render text.
    for (size_t i = 0; i < indices.size(); ++i)
    {
        RotatedRect& box = boxes[indices[i]];

        Point2f vertices[4];
        box.points(vertices);

        for (int j = 0; j < 4; ++j)
        {
            vertices[j].x *= ratio.x;
            vertices[j].y *= ratio.y;
        }

        if (!this->modelRecognition.empty())
        {
            Mat cropped;
            this->fourPointsTransform(frame, vertices, cropped);

            cvtColor(cropped, cropped, cv::COLOR_BGR2GRAY);

            Mat blobCrop = blobFromImage(cropped, 1.0 / 127.5, Size(), Scalar::all(127.5));
            this->recognizer.setInput(blobCrop);

            tickMeter.start();
            Mat result = this->recognizer.forward();
            tickMeter.stop();

            std::string wordRecognized = "";
            this->decodeText(result, wordRecognized);
            putText(frame, wordRecognized, vertices[1], FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255));
        }

        for (int j = 0; j < 4; ++j)
            line(frame, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 1);
    }
    // Put efficiency information.
    std::string label = format("Inference time: %.2f ms", tickMeter.getTimeMilli());
    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    tickMeter.reset();
}

int main()
{
    OCR ocr_model("CRNN_VGG_BiLSTM_CTC.onnx", "0123456789abcdefghijklmnopqrstuvwxyz");
    string imgpath = "sign.jpg";
    Mat srcimg = imread(imgpath);
    ocr_model.detect_rec(srcimg);

    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);
    imshow(kWinName, srcimg);
    waitKey(0);
    destroyAllWindows();
}