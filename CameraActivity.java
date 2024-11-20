package com.ece420.lab6;
import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.os.Bundle;
//import android.support.v7.app.AppCompatActivity;
// import android.util.Log;
import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.TextView;

import java.io.IOException;
import java.lang.Math;
// import java.util.List;


public class CameraActivity extends Activity implements SurfaceHolder.Callback{

    // UI Variable
    private SurfaceView surfaceView;
    private SurfaceHolder surfaceHolder;
    private SurfaceView surfaceView2;
    private SurfaceHolder surfaceHolder2;
    private TextView textHelper;
    // Camera Variable
    private Camera camera;
    boolean previewing = false;
    private int width = 640;
    private int height = 480;
    // Kernels
    private double[][] kernelS = new double[][] {{-1,-1,-1},{-1,9,-1},{-1,-1,-1}};
    private double[][] kernelX = new double[][] {{1,0,-1},{1,0,-1},{1,0,-1}};
    private double[][] kernelY = new double[][] {{1,1,1},{0,0,0},{-1,-1,-1}};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFormat(PixelFormat.UNKNOWN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_camera);
        super.setRequestedOrientation (ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);

        // Modify UI Text
        textHelper = (TextView) findViewById(R.id.Helper);
        if(MainActivity.appFlag == 1) textHelper.setText("Histogram Equalized Image");
        else if(MainActivity.appFlag == 2) textHelper.setText("Sharpened Image");
        else if(MainActivity.appFlag == 3) textHelper.setText("Edge Detected Image");

        // Setup Surface View handler
        surfaceView = (SurfaceView)findViewById(R.id.ViewOrigin);
        surfaceHolder = surfaceView.getHolder();
        surfaceHolder.addCallback(this);
        surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
        surfaceView2 = (SurfaceView)findViewById(R.id.ViewHisteq);
        surfaceHolder2 = surfaceView2.getHolder();
    }


    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        // Must have to override native method
        return;
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        if(!previewing) {
            camera = Camera.open();
            if (camera != null) {
                try {
                    // Modify Camera Settings
                    Camera.Parameters parameters = camera.getParameters();
                    parameters.setPreviewSize(width, height);
                    // Following lines could log possible camera resolutions, including
                    // 2592x1944;1920x1080;1440x1080;1280x720;640x480;352x288;320x240;176x144;
                    // List<Camera.Size> sizes = parameters.getSupportedPictureSizes();
                    // for(int i=0; i<sizes.size(); i++) {
                    //     int height = sizes.get(i).height;
                    //     int width = sizes.get(i).width;
                    //     Log.d("size: ", Integer.toString(width) + ";" + Integer.toString(height));
                    // }
                    camera.setParameters(parameters);
                    camera.setDisplayOrientation(90);
                    camera.setPreviewDisplay(surfaceHolder);
                    camera.setPreviewCallback(new PreviewCallback() {
                        public void onPreviewFrame(byte[] data, Camera camera)
                        {
                            // Lock canvas
                            Canvas canvas = surfaceHolder2.lockCanvas(null);
                            // Where Callback Happens, camera preview frame ready
                            onCameraFrame(canvas,data);
                            // Unlock canvas
                            surfaceHolder2.unlockCanvasAndPost(canvas);
                        }
                    });
                    camera.startPreview();
                    previewing = true;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        // Cleaning Up
        if (camera != null && previewing) {
            camera.stopPreview();
            camera.setPreviewCallback(null);
            camera.release();
            camera = null;
            previewing = false;
        }
    }

    // Camera Preview Frame Callback Function
    protected void onCameraFrame(Canvas canvas, byte[] data) {

        Matrix matrix = new Matrix();
        matrix.postRotate(90);
        int retData[] = new int[width * height];

        // Apply different processing methods
        if(MainActivity.appFlag == 1){
            byte[] histeqData = histEq(data, width, height);
            retData = yuv2rgb(histeqData);
        }
//        else if (MainActivity.appFlag == 2){
//
//            int[] sharpData = conv2(data, width, height, kernelS);
//            retData = merge(sharpData, sharpData);
//        }
//        else if (MainActivity.appFlag == 3){
////            PRE PROCESSING
//
//            int[] xData = conv2(data, width, height, kernelX);
//            int[] yData = conv2(data, width, height, kernelY);
//            retData = merge(xData, yData);
//        }
        else if (MainActivity.appFlag == 3) {
            // Step 1: Convert to grayscale
            int[] grayData = yuv2gray(data);

            // Step 2: Apply a blur
            int[] blurredData = applyBlur(grayData);
//            int[] blurredData = grayData;

            // Step 3: Apply thresholding to create a binary image
            int[] binaryData = applyThreshold(blurredData, 50); // Threshold value can be adjusted

            int[] dilatedData = applyDilation(binaryData, width, height);


            // Step 4: Perform edge detection using kernels
            int[] xData = conv2(dilatedData, width, height, kernelX);
            int[] yData = conv2(dilatedData, width, height, kernelY);

            // Merge X and Y edges
            retData = merge(xData, yData);
        }


        // Create ARGB Image, rotate and draw
        Bitmap bmp = Bitmap.createBitmap(retData, width, height, Bitmap.Config.ARGB_8888);
        bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true);
        canvas.drawBitmap(bmp, new Rect(0,0, height, width), new Rect(0,0, canvas.getWidth(), canvas.getHeight()),null);

        if (MainActivity.appFlag == 3) {
            detectAndDrawContours(canvas, retData);
        }
    }

    // Helper function to convert YUV to RGB
    public int[] yuv2rgb(byte[] data){
        final int frameSize = width * height;
        int[] rgb = new int[frameSize];

        for (int j = 0, yp = 0; j < height; j++) {
            int uvp = frameSize + (j >> 1) * width, u = 0, v = 0;
            for (int i = 0; i < width; i++, yp++) {
                int y = (0xff & ((int) data[yp])) - 16;
                y = y<0? 0:y;

                if ((i & 1) == 0) {
                    v = (0xff & data[uvp++]) - 128;
                    u = (0xff & data[uvp++]) - 128;
                }

                int y1192 = 1192 * y;
                int r = (y1192 + 1634 * v);
                int g = (y1192 - 833 * v - 400 * u);
                int b = (y1192 + 2066 * u);

                r = r<0? 0:r;
                r = r>262143? 262143:r;
                g = g<0? 0:g;
                g = g>262143? 262143:g;
                b = b<0? 0:b;
                b = b>262143? 262143:b;

                rgb[yp] = 0xff000000 | ((r << 6) & 0xff0000) | ((g >> 2) & 0xff00) | ((b >> 10) & 0xff);
            }
        }
        return rgb;
    }

    // Helper function to merge the results and convert GrayScale to RGB
    public int[] merge(int[] xdata,int[] ydata){
        int size = height * width;
        int[] mergeData = new int[size];
        for(int i=0; i<size; i++)
        {
            int p = (int)Math.sqrt((xdata[i] * xdata[i] + ydata[i] * ydata[i]) / 2);
            mergeData[i] = 0xff000000 | p<<16 | p<<8 | p;
        }
        return mergeData;
    }

    // Function for Histogram Equalization
    public byte[] histEq(byte[] data, int width, int height){
        byte[] histeqData = new byte[data.length];
        int size = height * width;


        // Perform Histogram Equalization
        // Note that you only need to manipulate data[0:size] that corresponds to luminance
        // The rest data[size:data.length] is for colorness that we handle for you
        // *********************** START YOUR CODE HERE  **************************** //
        int[] histogram = new int[256];
        int[] cdf = new int[256];

        for (int i = 0; i < size; i++) {
            if (data[i] < 0) {
                histogram[data[i]+256]++;
            }
            else
                histogram[data[i]]++;
        }

        for (int i = 0; i < histogram.length; i++) {
            if (i == 0) {
                cdf[i] = histogram[i];
            } else {
                cdf[i] = cdf[i - 1] + histogram[i];
            }
        }

        for (int i = 0; i < histogram.length; i++) {
            cdf[i] = (255*(cdf[i]-cdf[0]))/(size-1);
        }

        for (int i = 0; i < size; i++) {
            if (data[i] < 0) {
                histeqData[i] =(byte)cdf[data[i]+256];
            }
            else
                histeqData[i] = (byte) cdf[data[i]];
        }

        // *********************** End YOUR CODE HERE  **************************** //
        // We copy the colorness part for you, do not modify if you want rgb images
        for(int i=size; i<data.length; i++){
            histeqData[i] = data[i];
        }
        return histeqData;
    }

//    START NEW CODE

    public int[] applyThreshold(int[] grayData, int threshold) {
        int size = width * height;
        int[] binary = new int[size];

        for (int i = 0; i < size; i++) {
            binary[i] = grayData[i] >= threshold ? 255 : 0;
        }
        return binary;
    }

    public int[] applyDilation(int[] data, int width, int height) {
        int[] dilatedData = new int[width * height];
        int[][] structuringElement = {
                {1, 1, 1},
                {1, 1, 1},
                {1, 1, 1}
        };

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int maxVal = 0;
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        int neighborIndex = (y + i) * width + (x + j);
                        maxVal = Math.max(maxVal, data[neighborIndex]);
                    }
                }
                dilatedData[y * width + x] = maxVal;
            }
        }
        return dilatedData;
    }


    public int[] applyBlur(int[] data) {
        double[][] blurKernel = {
                {1/16.0, 2/16.0, 1/16.0},
                {2/16.0, 4/16.0, 2/16.0},
                {1/16.0, 2/16.0, 1/16.0}
        }; // A Gaussian blur kernel, less aggressive
        return conv2(data, width, height, blurKernel);
    }




    public int[] yuv2gray(byte[] data) {
        final int frameSize = width * height;
        int[] gray = new int[frameSize];

        for (int j = 0, yp = 0; j < height; j++) {
            for (int i = 0; i < width; i++, yp++) {
                int y = (0xff & ((int) data[yp])) - 16;
                y = y < 0 ? 0 : y;
                gray[yp] = y;
            }
        }
        return gray;
    }

    private void detectAndDrawContours(Canvas canvas, int[] edgeData) {
        android.graphics.Paint paint = new android.graphics.Paint();
        paint.setStyle(android.graphics.Paint.Style.STROKE);
        paint.setColor(android.graphics.Color.RED);
        paint.setStrokeWidth(5);

        boolean[] visited = new boolean[width * height];
        int edgeThreshold = 100; // Intensity threshold for edge detection
        int minContourArea = 2000; // Minimum area for valid bounding boxes

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;

                if (!visited[index] && (edgeData[index] & 0xFF) > edgeThreshold) {
                    Rect boundingBox = calculateBoundingBox(edgeData, visited, x, y, edgeThreshold);

                    if (boundingBox != null) {
                        int boxWidth = boundingBox.width();
                        int boxHeight = boundingBox.height();

                        if (boxWidth * boxHeight >= minContourArea) {
                            // Adjust for screen rendering
                            canvas.drawRect(boundingBox, paint);
                        }
                    }
                }
            }
        }
    }




    private Rect calculateBoundingBox(int[] edgeData, boolean[] visited, int startX, int startY, int threshold) {
        int minX = startX, maxX = startX, minY = startY, maxY = startY;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;

                if (!visited[index] && (edgeData[index] & 0xFF) > threshold) {
                    visited[index] = true;

                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        if (minX < maxX && minY < maxY) {
            return new Rect(minX, minY, maxX, maxY);
        }
        return null;
    }










//    END NEW CODE

//    public int[] conv2(byte[] data, int width, int height, double kernel[][]){
//        // 0 is black and 255 is white.
//        int size = height * width;
//        int[] convData = new int[size];
//
//        // Perform single channel 2D Convolution
//        // Note that you only need to manipulate data[0:size] that corresponds to luminance
//        // The rest data[size:data.length] is ignored since we only want grayscale output
//        // *********************** START YOUR CODE HERE  **************************** //
//
//        double[][] newkern = new double[kernel.length][kernel[0].length];
//
//        for (int i = 0; i < newkern.length; i++) {
//            for (int j = 0; j < newkern[0].length; j++) {
//                newkern[i][j] = kernel[newkern.length - 1 - i][newkern[0].length - 1 - j];
//            }
//        }
//
//        for (int col = 0; col < width; col++) {
//            for (int row = 0; row < height; row++) {
//                double summ = 0;
//                for (int i = 0; i < newkern.length; i++) {
//                    for (int j = 0; j < newkern[0].length; j++) {
//                        int kernx = j-newkern[0].length/2;
//                        int kerny = i-newkern.length/2;
//                        if ((kernx+col) >= 0 && (kernx+col) < width && (kerny+row) >= 0 && (kerny+row) < height) {
//                            summ += newkern[i][j] * (double)(data[(kerny+row) * width + (kernx+col)]);
//                        }
//                    }
//                }
//                convData[(row*width)+col] = (byte)(summ);
//            }
//        }
//        // *********************** End YOUR CODE HERE  **************************** //
//        return convData;
//    }
    public int[] conv2(int[] data, int width, int height, double[][] kernel) {
        int size = height * width;
        int[] convData = new int[size];

        // Flip the kernel
        double[][] newkern = new double[kernel.length][kernel[0].length];
        for (int i = 0; i < newkern.length; i++) {
            for (int j = 0; j < newkern[0].length; j++) {
                newkern[i][j] = kernel[newkern.length - 1 - i][newkern[0].length - 1 - j];
            }
        }

        // Perform Convolution
        for (int col = 0; col < width; col++) {
            for (int row = 0; row < height; row++) {
                double summ = 0.0;
                for (int i = 0; i < newkern.length; i++) {
                    for (int j = 0; j < newkern[0].length; j++) {
                        int kernx = j - newkern[0].length / 2;
                        int kerny = i - newkern.length / 2;

                        // Ensure kernel doesn't go out of bounds
                        if ((kernx + col) >= 0 && (kernx + col) < width &&
                                (kerny + row) >= 0 && (kerny + row) < height) {
                            int index = (kerny + row) * width + (kernx + col);
                            summ += newkern[i][j] * data[index];
                        }
                    }
                }
                // Clamp the result to valid grayscale range [0, 255]
                convData[row * width + col] = Math.max(0, Math.min(255, (int)summ));
            }
        }
        return convData;
    }



}
