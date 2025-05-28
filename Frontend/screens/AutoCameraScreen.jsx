// ManualCameraScreen.jsx
import React, { useEffect, useRef, useState } from 'react';
import { View, Text, StyleSheet, Button, TouchableOpacity, ActivityIndicator } from 'react-native';
import { CameraView, Camera } from 'expo-camera';
import uuid from 'react-native-uuid';

export default function ManualCameraScreen({ navigation, items, setItems }) {
    const cameraRef = useRef(null);
    const [hasPermission, setHasPermission] = useState(null);
    const [isCameraReady, setIsCameraReady] = useState(false);
    const [isCapturing, setIsCapturing] = useState(false);

    useEffect(() => {
        (async () => {
            const { status } = await Camera.requestCameraPermissionsAsync();
            setHasPermission(status === 'granted');
        })();
    }, []);

    const analyzeImage = async (photoUri, itemId) => {
        try {
            const formData = new FormData();
            formData.append('image', {
                uri: photoUri,
                type: 'image/jpeg',
                name: 'photo.jpg',
            });

            const response = await fetch('http://192.168.107.100:500/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                body: formData,
            });

            const data = await response.json();
            const predictedClass = data?.predicted_class || 'Bilinmiyor';

            setItems((prev) =>
                prev.map(item =>
                    item.id === itemId
                        ? { ...item, status: predictedClass }
                        : item
                )
            );
        } catch (err) {
            console.error('Backend analiz hatası:', err);
            setItems((prev) =>
                prev.map(item =>
                    item.id === itemId
                        ? { ...item, status: 'Analiz Hatası' }
                        : item
                )
            );
        }
    };

    const handleTakePicture = async () => {
        if (!cameraRef.current || isCapturing) return;
        setIsCapturing(true);
        try {
            const photo = await cameraRef.current.takePictureAsync({ quality: 0.5 });

            const newItemId = uuid.v4().toString();
            const newItem = {
                id: newItemId,
                imageUri: photo.uri,
                status: 'Analiz Ediliyor...',
            };
            setItems((prev) => [newItem, ...prev]);
            analyzeImage(photo.uri, newItemId);
        } catch (err) {
            console.error('Fotoğraf çekim hatası:', err);
        } finally {
            setIsCapturing(false);
        }
    };

    if (hasPermission === null) {
        return <View><Text>İzin kontrol ediliyor...</Text></View>;
    }

    if (hasPermission === false) {
        return <View><Text>Kamera izni verilmedi.</Text></View>;
    }

    return (
        <View style={styles.container}>
            <CameraView
                ref={cameraRef}
                style={styles.camera}
                facing="back"
                onCameraReady={() => setIsCameraReady(true)}
            />
            <View style={styles.overlay}>
                <Text style={styles.overlayText}>Manuel Kamera Modu</Text>
                <TouchableOpacity
                    style={styles.captureButton}
                    onPress={handleTakePicture}
                    disabled={!isCameraReady || isCapturing}
                >
                    {isCapturing ? (
                        <ActivityIndicator color="#fff" />
                    ) : (
                        <Text style={styles.buttonText}>Fotoğraf Çek</Text>
                    )}
                </TouchableOpacity>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    camera: {
        flex: 1,
    },
    overlay: {
        position: 'absolute',
        bottom: 40,
        alignSelf: 'center',
        alignItems: 'center',
        backgroundColor: 'rgba(0,0,0,0.6)',
        padding: 20,
        borderRadius: 16,
    },
    overlayText: {
        color: '#fff',
        fontSize: 17,
        marginBottom: 12,
        fontWeight: 'bold'
    },
    captureButton: {
        backgroundColor: '#2196F3',
        borderRadius: 40,
        paddingVertical: 14,
        paddingHorizontal: 32,
        alignItems: 'center',
        marginTop: 10,
    },
    buttonText: {
        color: '#fff',
        fontSize: 18,
        fontWeight: '600',
    },
});
