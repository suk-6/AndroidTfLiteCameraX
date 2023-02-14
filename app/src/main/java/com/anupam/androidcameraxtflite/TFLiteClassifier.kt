package com.anupam.androidcameraxtflite

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks.call
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.*
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.collections.ArrayList
import android.graphics.Bitmap
import android.os.*
import org.tensorflow.lite.Interpreter
import android.speech.tts.TextToSpeech

class TFLiteClassifier(private val context: Context) {

    private var interpreter: Interpreter? = null
    var isInitialized = false
        private set

    private var gpuDelegate: GpuDelegate? = null

    var labels = ArrayList<String>()

    private val executorService: ExecutorService = Executors.newCachedThreadPool()

    private var inputImageWidth: Int = 0
    private var inputImageHeight: Int = 0
    private var modelInputSize: Int = 0

    private var tts: TextToSpeech? = null

//    private fun initTextToSpeech() {
//        Log.d(TAG, "initTextToSpeech: 함수실행")
//        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP){
//            return
//        }
//        tts = TextToSpeech(context, TextToSpeech.OnInitListener {
//            if(it == TextToSpeech.SUCCESS){
//                var result = tts?.setLanguage(Locale.ENGLISH)
//                if(result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED){
//                    return@OnInitListener
//                }
//            }
//        })
//    }

    fun initialize(): Task<Void> {
        Log.d(TAG, "initTextToSpeech: 함수실행")
        tts = TextToSpeech(context, TextToSpeech.OnInitListener {
            if(it == TextToSpeech.SUCCESS){
                var result = tts?.setLanguage(Locale.ENGLISH)
                if(result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED){
                    return@OnInitListener
                }
            }
        })
        return call(
            executorService,
            Callable<Void> {
                initializeInterpreter()
                null
            }
        )
    }
    @Throws(IOException::class)
    private fun initializeInterpreter() {



        val assetManager = context.assets
        val model = loadModelFile(assetManager, "mobilenet_v1_1.0_224.tflite")

        labels = loadLines(context, "labels.txt")
        val options = Interpreter.Options()
        gpuDelegate = GpuDelegate()
        options.addDelegate(gpuDelegate)
        val interpreter = Interpreter(model, options)

        val inputShape = interpreter.getInputTensor(0).shape()
        inputImageWidth = inputShape[1]
        inputImageHeight = inputShape[2]
        modelInputSize = FLOAT_TYPE_SIZE * inputImageWidth * inputImageHeight * CHANNEL_SIZE

        this.interpreter = interpreter

        isInitialized = true
    }

    @Throws(IOException::class)
    private fun loadModelFile(assetManager: AssetManager, filename: String): ByteBuffer {
        val fileDescriptor = assetManager.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    @Throws(IOException::class)
    fun loadLines(context: Context, filename: String): ArrayList<String> {
        val s = Scanner(InputStreamReader(context.assets.open(filename)))
        val labels = ArrayList<String>()
        while (s.hasNextLine()) {
            labels.add(s.nextLine())
        }
        s.close()
        return labels
    }

    private fun getMaxResult(result: FloatArray): Int {
        var probability = result[0]
        var index = 0
        for (i in result.indices) {
            if (probability < result[i]) {
                probability = result[i]
                index = i
            }
        }
        return index
    }



    private fun classify(bitmap: Bitmap): String {

        check(isInitialized) { "TF Lite Interpreter is not initialized yet." }
        val resizedImage =
            Bitmap.createScaledBitmap(bitmap, inputImageWidth, inputImageHeight, true)

        val byteBuffer = convertBitmapToByteBuffer(resizedImage)

        val output = Array(1) { FloatArray(labels.size) }
        val startTime = SystemClock.uptimeMillis()
        interpreter?.run(byteBuffer, output)
        val endTime = SystemClock.uptimeMillis()

        var inferenceTime = endTime - startTime
        var index = getMaxResult(output[0])
        var result = "${labels[index]}\nInference Time $inferenceTime ms"

//        ttsSpeak(labels[index])
        countSpeak(index)

        return result
    }
    var speakArray = Array<Int>(3) { -1 }

    private fun countSpeak(index: Int) {
        if (speakArray[0] == -1) {
            speakArray[0] = index
        }
        else if (speakArray[1] == -1) {
            speakArray[1] = index
        }
        else if (speakArray[2] == -1) {
            speakArray[2] = index
        }
        else {
            if (speakArray[0] == speakArray[1]) {
                if (speakArray[1] == speakArray[2]) {
                    ttsSpeak(labels[speakArray[0]])
                }
            }

            speakArray[0] = index
            speakArray[1] = -1
            speakArray[2] = -1
        }
    }

    private fun ttsSpeak(strTTS: String) {
        tts?.speak(strTTS, TextToSpeech.QUEUE_FLUSH, null, null)
//        tts?.playSilentUtterance(1000, TextToSpeech.QUEUE_ADD, null)
    }

    fun classifyAsync(bitmap: Bitmap): Task<String> {
        Thread.sleep(250)
        return call(executorService, Callable<String> { classify(bitmap) })
    }

    fun close() {
        if (tts != null) {
            tts?.stop();
            tts?.shutdown();
        }
        call(
            executorService,
            Callable<String> {
                interpreter?.close()
                if (gpuDelegate != null) {
                    gpuDelegate!!.close()
                    gpuDelegate = null
                }

                Log.d(TAG, "Closed TFLite interpreter.")
                null
            }
        )
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(modelInputSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputImageWidth * inputImageHeight)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until inputImageWidth) {
            for (j in 0 until inputImageHeight) {
                val pixelVal = pixels[pixel++]

                byteBuffer.putFloat(((pixelVal shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                byteBuffer.putFloat(((pixelVal shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                byteBuffer.putFloat(((pixelVal and 0xFF) - IMAGE_MEAN) / IMAGE_STD)

            }
        }
        bitmap.recycle()

        return byteBuffer
    }

    companion object {
        private const val TAG = "TfliteClassifier"
        private const val FLOAT_TYPE_SIZE = 4
        private const val CHANNEL_SIZE = 3
        private const val IMAGE_MEAN = 127.5f
        private const val IMAGE_STD = 127.5f
    }
}