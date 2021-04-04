#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <opencv2\opencv.hpp>
#include <vector>
#include <fstream>
#include <windows.h>
#include <cmath>
#include <time.h>


using namespace std;
using namespace cv;


bool TrainingSwitch = false;
bool Circle_Square = false; //True for circles
bool GeneratorSwitch = false; //True for turning off then generation is complite

const int iSize_S = 10;
const int iSize_N = 300;
const int iSeze_B = 1024;

int iCoords_X[iSize_N][2];
int iCoords_Y[iSize_N][2];

int iFernDesc[iSize_N/iSize_S][iSeze_B];
float iClassifier_C[iSize_N / iSize_S][iSeze_B];
float iClassifier_S[iSize_N / iSize_S][iSeze_B];

const int iImCount = 2500;
const int iImSize_X = 30;
const int iCircleR = 8;

vector<Mat> savedImages;
vector<vector<bool>> descriptorTable;
Mat Descriptor;

bool makeImage() {

    srand((unsigned)time(NULL));
    
    int iPCoord_X;
    int iPCoord_Y;
    
    //string classof = NULL;

    for (size_t i = 0; i < iImCount; i++)
    {

        Mat mImgFeed = Mat::zeros(iImSize_X, iImSize_X, CV_8UC3);

        while ((iPCoord_X = rand()) < 8 || iPCoord_X > 22)
        { iPCoord_X /= 2; }

        while ((iPCoord_Y = rand()) < 8 || iPCoord_Y > 22)
        { iPCoord_Y /= 2; }

        Point pCenter = Point(iPCoord_X, iPCoord_Y);
        Point pCorner_U = Point(iPCoord_X+7, iPCoord_Y+7);
        Point pCorner_D = Point(iPCoord_X-7, iPCoord_Y-7);
        if (Circle_Square)
        {
            circle(mImgFeed, pCenter, iCircleR, Scalar(255, 255, 255), FILLED, LINE_8);
        }
        else {
            rectangle(mImgFeed, pCorner_U, pCorner_D, Scalar(255, 255, 255), FILLED, LINE_8);
        }

        //if (Circle_Square) {
        //    classof = "C";
        //}
        //else {
        //    classof = "S";
        //}

        string savingName = "D://TrainingImages/" + to_string(i) + ".jpg";
        imwrite(savingName, mImgFeed);

    }

    return true;
}

bool makeCoordTable() {

    srand((unsigned)time(NULL));
    int iPCoord;

    ofstream outStream("D:\\TokenCoords\\TokenCoords.txt");
    if (outStream)
    {

        for (size_t c = 0; c < (iSize_N*2); c++) {

            for (size_t i = 0; i < 2; i++)
            {
                iPCoord = rand();
                while (iPCoord > 30)
                {
                    iPCoord %= 30;
                }

                outStream << iPCoord << endl;

            }
        }

        outStream.close();
        return true;

    }

    return false;

}

int loadTrainingImageRow(vector<Mat>& savedImages) {

    cout << "Start to load images" << endl;

    WIN32_FIND_DATA FindFileData;

    HANDLE hFindTraining;
    int iNumOfIMG = 0;

        hFindTraining = FindFirstFile(L"D:\\TrainingImages\\*.jpg", &FindFileData);
        if (hFindTraining != NULL) {
            do {

                wstring wFindFileData = FindFileData.cFileName;
                savedImages.push_back(imread("D:\\TrainingImages\\" + string(wFindFileData.begin(), wFindFileData.end()), IMREAD_GRAYSCALE));
                iNumOfIMG++;
            } while (FindNextFile(hFindTraining, &FindFileData) != 0);
        }
        else
        {
            cout << "Found nothing" << endl;
        }

    return iNumOfIMG;
}

int loadExperimentalImageRow(vector<Mat>& savedImages) {

    cout << "Start to load images" << endl;

    WIN32_FIND_DATA FindFileData;

    HANDLE hFindExperimental;
    int iNumOfIMG = 0;

        hFindExperimental = FindFirstFile(L"D:\\ExperimentalImages\\*.jpg", &FindFileData);


        if (hFindExperimental != NULL) {
            do {

                wstring wFindFileData = FindFileData.cFileName;
                savedImages.push_back(imread("D:\\ExperimentalImages\\" + string(wFindFileData.begin(), wFindFileData.end()), IMREAD_GRAYSCALE));

                iNumOfIMG++;
            } while (FindNextFile(hFindExperimental, &FindFileData) != 0);
        }
        else
        {
            cout << "Found nothing" << endl;
        }

    return iNumOfIMG;
}

bool loadDescriptorCoord() {

    cout << "Start to load coordinates" << endl;

    WIN32_FIND_DATA FindFileData;
    HANDLE hFind;

    hFind = FindFirstFile(L"D:\\TokenCoords\\*.txt", &FindFileData);

    if (hFind != NULL) {

        wstring wFindFileData = FindFileData.cFileName;
        string sFileNamePath = "D:\\TokenCoords\\" + string(wFindFileData.begin(), wFindFileData.end());

            ifstream inStream(sFileNamePath);
            if (inStream) {

                for (int iRow = 0; iRow < iSize_N; iRow++) {
                    for (int iCol = 0; iCol < 2; iCol++) {

                        float read;
                        inStream >> read;

                        iCoords_X[iRow][iCol] = read;
                    }
                }

                for (int iRow = 0; iRow < iSize_N; iRow++) {
                    for (int iCol = 0; iCol < 2; iCol++) {

                        float read;
                        inStream >> read;

                        iCoords_Y[iRow][iCol] = read;
                    }
                }
            }
            inStream.close();
            return true;
    }
    else
    {
        cout << "Found nothing" << endl;
        return false;
    }

}

bool imageProsessing(Mat& savedImages) {
    vector<bool> bDescriptor;
    Descriptor = Mat::zeros(iImSize_X, iImSize_X, CV_8UC3);;
    for (int iRow = 0; iRow < iSize_N; iRow++) {


        int value1 = (int)savedImages.at<uchar>(iCoords_X[iRow][0], iCoords_X[iRow][1]);
        int value2 = (int)savedImages.at<uchar>(iCoords_Y[iRow][0], iCoords_Y[iRow][1]);

        //line(Descriptor, Point(iCoords_X[iRow][0], iCoords_X[iRow][1]), Point(iCoords_Y[iRow][0], iCoords_Y[iRow][1]), Scalar(255, 255, 255), 1, LINE_8);
            if (value2 > value1) {
                bDescriptor.push_back(true);
            }
            else {
                bDescriptor.push_back(false);
            }

    }
    descriptorTable.push_back(bDescriptor);
    bDescriptor.clear();
    return true;
}

int* descriptorAnalyze(vector<bool> bDescriptor) {

    int counter;
    int iCounter_L = 0;
    for (size_t L = 0; L < iSize_N; L)
    {
        counter = 0;
        for (size_t S = 0; S < iSize_S; S++) {
            if (bDescriptor[L + S]) {
                counter += pow(2, (iSize_S - 1) - S);
            }
        }

        iFernDesc[iCounter_L][counter]++;
        L += iSize_S;
        iCounter_L++;
    }
    iCounter_L;
    return *iFernDesc;
}

bool loadDescriptorCircle() {

    cout << "Start to load Circles" << endl;

        string sCirclePath = "D:\\FernDescriptor\\CircleClass.txt";
        int iCheck = 0;
        ifstream inStream(sCirclePath);
        if (inStream) {

            for (size_t c = 0; c < (iSize_N / iSize_S); c++) {
                for (size_t i = 0; i < iSeze_B; i++)
                {
                    int read;
                    inStream >> read;

                    iClassifier_C[c][i] = read;

                    iCheck += read;
                }
            }
            inStream.close();

            if (iCheck == ((iSize_N / iSize_S) * iImCount)) {
                return true;
            }
        }
        else
        {
            cout << "Load error #" << iCheck << endl;
            return false;
        }
}

bool loadDescriptorSquare() {

    cout << "Start to load Squares" << endl;
    int iCheck = 0;
    string sSquarePath = "D:\\FernDescriptor\\SquareClass.txt";

    ifstream inStream(sSquarePath);
    if (inStream) {

        for (size_t c = 0; c < (iSize_N / iSize_S); c++) {
            for (size_t i = 0; i < iSeze_B; i++)
            {
                int read;
                inStream >> read;

                iClassifier_S[c][i] = read;
                iCheck += read;
            }
        }
        inStream.close();
        if (iCheck == ((iSize_N / iSize_S) * iImCount)) {
            return true;
        }
    }
    else
    {
        cout << "Load error #" << iCheck << endl;
        return false;
    }
}

//функция создания файла
bool saveСlassifier(int(*iFernDesc)[iSeze_B], int iNumOfIMG) {

    string sPathFileName;
    if (Circle_Square)
    {
        sPathFileName = "D:\\FernDescriptor\\CircleClass.txt";
    }
    else {
        sPathFileName = "D:\\FernDescriptor\\SquareClass.txt";
    }
    ofstream outStream(sPathFileName);
    if (outStream)
    {
        for (size_t c = 0; c < (iSize_N / iSize_S); c++) {
            for (size_t i = 0; i < iSeze_B; i++)
            {
                float value;
                value = iFernDesc[c][i];
                outStream << value << endl;
                iFernDesc[c][i] = 0;
            }
            outStream << endl;
        }

        outStream.close();
        return true;

    }
    return false;
}

bool classifier(int i, vector<bool> bDescriptor) {

    long float fProbability_C = 1;
    long float fProbability_S = 1;

    int counter;
    int iCounter_L = 0;

    for (size_t L = 0; L < iSize_N; L)
    {
        counter = 0;
        for (size_t S = 0; S < iSize_S; S++) {
            if (bDescriptor[L + S]) {
                counter += pow(2, (iSize_S - 1) - S);
            }
        }

        if (iClassifier_C[iCounter_L][counter] != 0) {
            fProbability_C = fProbability_C * iClassifier_C[iCounter_L][counter];
        }

        L += iSize_S;
        iCounter_L++;
    }

    iCounter_L = 0;

    for (size_t L = 0; L < iSize_N; L)
    {
        counter = 0;
        for (size_t S = 0; S < iSize_S; S++) {
            if (bDescriptor[L + S]) {
                counter += pow(2, (iSize_S - 1) - S);
            }
        }

        if (iClassifier_S[iCounter_L][counter] != 0) {
            fProbability_S = fProbability_S * iClassifier_S[iCounter_L][counter];
        }

        L += iSize_S;
        iCounter_L++;
    }


    cout << fProbability_C << "_ _" << fProbability_S << endl;
    if (fProbability_C > fProbability_S) {
        return 1; 
    }
    else if (fProbability_C == fProbability_S || (fProbability_S * fProbability_C) == 0) {
        cout << "Classifier Error" << endl;
        return 0;
    }
    else {
        return 0;
    }

    }


int main(int argc, TCHAR* argv[]) {

    namedWindow("CamWindow", WINDOW_AUTOSIZE);
    int iNumOfIMG;
    do {
        descriptorTable.clear();
        savedImages.clear();

        if (TrainingSwitch) {
            if (Circle_Square)
            {
                //makeCoordTable();
            }

            makeImage();

            if (GeneratorSwitch)
            {
                return 0;
            }

            iNumOfIMG = loadTrainingImageRow(savedImages);
        }
        else
        { 
            iNumOfIMG = loadExperimentalImageRow(savedImages); 
        }

        if (iNumOfIMG == 0) {
            std::cout << "Faled to load images!";
        }
        else
        {

            if (!loadDescriptorCoord()) {

                cout << "Failed to load descriptor coordinates!";
                return 0;

            }
            else
            {
                if (!TrainingSwitch) {
                    if (!loadDescriptorCircle()) {
                        cout << "Faled to load Circles!";
                        return 0;
                    }
                    if (!loadDescriptorSquare()) {
                        cout << "Faled to load Squares!";
                        return 0;
                    }
                }
                for (int i = 0; i < iNumOfIMG; i++) {
                    imageProsessing(savedImages[i]);
                    imshow("CamWindow", Descriptor);
                    waitKey(1);
                }
                if (TrainingSwitch) {
                    for (int i = 0; i < iNumOfIMG; i++) {
                        descriptorAnalyze(descriptorTable[i]);
                    }


                    if (!saveСlassifier(iFernDesc, iNumOfIMG))
                    {
                        cout << "Сlassifier Error!" << endl;
                    }
                    else {
                        cout << "Сlassifier for: ";
                        if (Circle_Square) {
                            cout << "Circles Ready!" << endl;
                            Circle_Square = false;

                        }
                        else { 
                            cout << "Squares Ready!" << endl; 
                            TrainingSwitch = false;

                        }
                    }
                }
                else {
                    int countofErrors = 0;
                    int countofCircles = 0;
                    int countofSquares = 0;
                    for (size_t i = 0; i < iNumOfIMG; i++)
                    {

                        if (classifier(i, descriptorTable[i])) {

                            cout << i << "_It's a Circle!_" << ++countofCircles << endl << endl;

                        }
                        else {

                            countofErrors++;
                            cout << i << "_It's a Square!_" << ++countofSquares << endl << endl;

                        }
                    }
                    if (countofErrors < 100) {

                        cout << countofErrors << endl;
                        return 0;

                    }
                    else {

                        cout << countofErrors << endl;
                        TrainingSwitch = true;
                        Circle_Square = true;
                        return 0;

                    }
                }
            }
        }
    } while (1);
}

