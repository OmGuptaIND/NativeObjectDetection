import { cameraWithTensors } from "@tensorflow/tfjs-react-native";
import { Camera } from "expo-camera";
import React, { useEffect, useRef, useState } from "react";
import {
    Dimensions,
    LogBox,
    Platform,
    StyleSheet,
    Text,
    View,
} from "react-native";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs";
import Canvas, { CanvasRenderingContext2D } from "react-native-canvas";

const TensorCamera = cameraWithTensors(Camera);
const { width, height } = Dimensions.get("window");

export default function App() {  
    const [model, setModel] = useState<cocoSsd.ObjectDetection>();
    let context = useRef<CanvasRenderingContext2D>();
    let canvas = useRef<Canvas>();    
    let textureDim =
        Platform.OS == "ios"
            ? { height: 1920, width: 1080 }
            : { height: 1200, width: 1600 };

    const handleCameraStream = (images: any) => {
        const loop = async () => {
            const nextImageTensor = images.next().value;
            if (!model || !nextImageTensor) {
                throw new Error("No Image or image Tensor ");
            }
            model
                .detect(nextImageTensor)
                .then((prediction) => {
                    //Draw Box here
                    console.log("this is prediction ", prediction);

                    drawRectangle( prediction, nextImageTensor );
                })
                .catch((err) => console.log(err));
        };
        loop();
    };

    useEffect(() => {
        const startBackend = async () => {
            const { status } = await Camera.requestCameraPermissionsAsync();
            console.log(status);
            await tf.ready();
            setModel(await cocoSsd.load());
        };
        startBackend();
    }, []);

    const drawRectangle = (
        predictions: cocoSsd.DetectedObject[],
        nextImageTensor: any
    ) => {
        if (!context.current || !canvas.current) return;

        const scaleWidth = width / nextImageTensor.shape[1];
        const scaleheight = height / nextImageTensor.shape[0];

        const flipHorizontalIfAndroid = Platform.OS == "ios" ? false : true;

        context.current.clearRect(0, 0, width, height);

        for (const prediction of predictions) {
            const [x, y, width, height] = prediction.bbox;
            const boundingBoxX = flipHorizontalIfAndroid
                ? canvas.current.width - x * scaleWidth - width * scaleWidth
                : x * scaleWidth;
            const boundingBoxY = y * scaleheight;

            context.current.strokeRect(
                boundingBoxX,
                boundingBoxY,
                width * scaleWidth,
                height * scaleheight
            );
            context.current.strokeText(
                prediction.class,
                boundingBoxX,
                boundingBoxY
            );
        }
    };

    const handleCanvas = (can: Canvas) => {
        if (can) {
            can.width = width;
            can.height = height;
            const ctx: CanvasRenderingContext2D = can.getContext("2d");
            ctx.strokeStyle = "red";
            ctx.fillStyle = "red";
            ctx.lineWidth = 3;

            context.current = ctx;
            canvas.current = can;
        }
    };
    return (
        <View style={styles.container}>
            <TensorCamera
                style={styles.camera}
                type={Camera.Constants.Type.back}
                cameraTextureHeight={textureDim.height}
                cameraTextureWidth={textureDim.width}
                resizeHeight={200}
                resizeWidth={152}
                resizeDepth={3}
                onReady={handleCameraStream}
                autorender={true}
                useCustomShadersToResize={false}
            />
            <Canvas style={styles.canvas} ref={handleCanvas} /> 
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: "#fff",
        alignItems: "center",
        justifyContent: "center",
    },
    camera: {
        width: "100%",
        height: "100%",
    },
    canvas: {
        position: "absolute",
        zIndex: 1000000,
        width: "100%",
        height: "100%",
    },
});
