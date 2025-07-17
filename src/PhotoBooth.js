import React, { useRef, useEffect, useState, useCallback } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import html2canvas from 'html2canvas';
import { initializeApp } from 'firebase/app';
import { getStorage, ref, uploadString, getDownloadURL } from 'firebase/storage';
import { QRCodeCanvas } from 'qrcode.react';
import { firebaseConfig } from './firebase-config.js';

// CSS for fun pink cartoon style with animations
const globalStyles = `
  @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
  
  body, html {
    margin: 0;
    padding: 0;
    height: 100%;
    overflow-x: hidden;
    font-family: 'Nunito', sans-serif;
    background: linear-gradient(135deg, #FFE4F0 0%, #FFB8E1 50%, #EE9ABF 100%);
  }
  
  #root {
    height: 100vh;
  }
  
  @media (max-width: 768px) {
    body {
      font-size: 14px;
    }
  }

  @keyframes pulse {
    0%, 100% { 
      transform: scale(1); 
      opacity: 1; 
    }
    50% { 
      transform: scale(1.15); 
      opacity: 0.85; 
    }
  }

  @keyframes bounce {
    0%, 20%, 50%, 80%, 100% {
      transform: translateY(0);
    }
    40% {
      transform: translateY(-35px);
    }
    60% {
      transform: translateY(-15px);
    }
  }

  @keyframes wiggle {
    0%, 100% { transform: rotate(0deg); }
    25% { transform: rotate(3deg); }
    75% { transform: rotate(-3deg); }
  }

  @keyframes loading {
    0% {
      transform: translateX(-100%);
      background: linear-gradient(90deg, #EE9ABF, #FFB8E1, #EE9ABF);
    }
    50% {
      transform: translateX(0%);
      background: linear-gradient(90deg, #FFB8E1, #EE9ABF, #FFB8E1);
    }
    100% {
      transform: translateX(100%);
      background: linear-gradient(90deg, #EE9ABF, #FFB8E1, #EE9ABF);
    }
  }

  @keyframes flash {
    0% { opacity: 0; }
    50% { opacity: 1; background: rgba(238, 154, 191, 0.8); }
    100% { opacity: 0; }
  }

  @keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
  }

  @keyframes sparkle {
    0%, 100% { transform: scale(0) rotate(0deg); opacity: 0; }
    50% { transform: scale(1) rotate(180deg); opacity: 1; }
  }
`;

// Inject global styles
if (typeof document !== 'undefined') {
    const styleSheet = document.createElement("style");
    styleSheet.type = "text/css";
    styleSheet.innerText = globalStyles;
    document.head.appendChild(styleSheet);
}

// Firebase configuration is imported from ./firebase-config.js
// Update that file with your actual Firebase project settings

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const storage = getStorage(app);

// Face Tracking Registry for Persistent Mask Assignment
class FaceRegistry {
    constructor() {
        this.trackedFaces = new Map(); // Map of faceId -> faceData
        this.nextFaceId = 1;
        this.maxFaces = 5;
        this.matchThreshold = 0.15; // Distance threshold for face matching
        this.maxFramesMissing = 1; // Remove face after missing for 1 frame (aggressive cleanup)
        this.maxFaceAge = 2000; // Remove faces older than 2 seconds without updates
    }

    // Calculate distance between two face centers
    calculateDistance(face1, face2) {
        const dx = face1.center.x - face2.center.x;
        const dy = face1.center.y - face2.center.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    // Calculate face similarity score
    calculateSimilarity(face1, face2) {
        // Distance-based similarity (closer = more similar)
        const distance = this.calculateDistance(face1, face2);
        const distanceScore = Math.max(0, 1 - (distance / this.matchThreshold));

        // Size-based similarity
        const sizeDiff = Math.abs(face1.size - face2.size) / Math.max(face1.size, face2.size);
        const sizeScore = Math.max(0, 1 - sizeDiff);

        // Combined similarity score
        return (distanceScore * 0.7) + (sizeScore * 0.3);
    }

    // Find best matching tracked face for a new detection
    findBestMatch(newFace) {
        let bestMatch = null;
        let bestScore = 0;

        for (const [faceId, trackedFace] of this.trackedFaces) {
            const similarity = this.calculateSimilarity(newFace, trackedFace);

            if (similarity > bestScore && similarity > 0.6) { // Minimum similarity threshold
                bestScore = similarity;
                bestMatch = { faceId, trackedFace, score: similarity };
            }
        }

        return bestMatch;
    }

    // Get next available mask number
    getNextMaskNumber() {
        const usedMasks = new Set();
        for (const [_, face] of this.trackedFaces) {
            usedMasks.add(face.maskNumber);
        }

        for (let i = 1; i <= this.maxFaces; i++) {
            if (!usedMasks.has(i)) {
                return i;
            }
        }
        return 1; // Fallback to mask 1 if all are used
    }

    // Process new face detections and assign masks
    processDetections(currentFaces) {
        const results = [];
        const matchedFaceIds = new Set();

        // Convert MediaPipe detections to our format
        const newDetections = currentFaces.map((landmarks, index) => {
            const center = this.calculateFaceCenter(landmarks);
            const size = this.calculateFaceSize(landmarks);
            const rotation = this.calculateHeadRotation(landmarks);

            return {
                landmarks,
                center,
                size,
                rotation,
                originalIndex: index
            };
        }).filter(face => face.center !== null);

        // Try to match each new detection with existing tracked faces
        for (const newFace of newDetections) {
            const match = this.findBestMatch(newFace);

            if (match) {
                // Update existing tracked face
                const { faceId, trackedFace } = match;
                trackedFace.center = newFace.center;
                trackedFace.size = newFace.size;
                trackedFace.rotation = newFace.rotation;
                trackedFace.landmarks = newFace.landmarks;
                trackedFace.framesMissing = 0;
                trackedFace.lastSeen = Date.now();

                matchedFaceIds.add(faceId);
                results.push({
                    ...newFace,
                    faceId,
                    maskNumber: trackedFace.maskNumber
                });
            } else {
                // New face detected - create new tracking entry
                const maskNumber = this.getNextMaskNumber();
                const faceId = `face_${this.nextFaceId++}`;

                this.trackedFaces.set(faceId, {
                    faceId,
                    maskNumber,
                    center: newFace.center,
                    size: newFace.size,
                    rotation: newFace.rotation,
                    landmarks: newFace.landmarks,
                    framesMissing: 0,
                    firstSeen: Date.now(),
                    lastSeen: Date.now()
                });

                results.push({
                    ...newFace,
                    faceId,
                    maskNumber
                });

                console.log(`🎭 New person detected! Assigned mask${maskNumber} (ID: ${faceId})`);
            }
        }

        // Update missing frame counts and cleanup old faces
        const facesToRemove = [];
        const currentTime = Date.now();
        
        for (const [faceId, trackedFace] of this.trackedFaces) {
            if (!matchedFaceIds.has(faceId)) {
                trackedFace.framesMissing++;

                if (trackedFace.framesMissing > this.maxFramesMissing) {
                    console.log(`👋 Person left frame - removing mask${trackedFace.maskNumber} (ID: ${faceId})`);
                    facesToRemove.push(faceId);
                }
            }
            
            // Also remove faces that are too old (stale faces)
            const faceAge = currentTime - trackedFace.lastSeen;
            if (faceAge > this.maxFaceAge) {
                console.log(`⏰ Removing stale mask${trackedFace.maskNumber} (ID: ${faceId}) - age: ${faceAge}ms`);
                facesToRemove.push(faceId);
            }
        }

        // Remove faces that have been missing too long
        facesToRemove.forEach(faceId => this.trackedFaces.delete(faceId));

        // If no faces detected at all, clear everything immediately
        if (currentFaces.length === 0) {
            if (this.trackedFaces.size > 0) {
                console.log('🧹 No faces detected - clearing all tracked faces immediately');
                this.trackedFaces.clear();
            }
            return [];
        }

        return results;
    }

    // Reset and clear all tracked faces
    reset() {
        if (this.trackedFaces.size > 0) {
            console.log('🔄 Resetting face registry - clearing all tracked faces');
            this.trackedFaces.clear();
        }
    }

    // Helper methods for face analysis (using same logic as existing code)
    calculateFaceCenter(landmarks) {
        if (!landmarks || landmarks.length === 0) return null;

        try {
            const leftEyeLeft = landmarks[33];
            const leftEyeRight = landmarks[133];
            const rightEyeLeft = landmarks[362];
            const rightEyeRight = landmarks[263];

            if (!leftEyeLeft || !leftEyeRight || !rightEyeLeft || !rightEyeRight) {
                return null;
            }

            const leftEyeCenter = {
                x: (leftEyeLeft.x + leftEyeRight.x) / 2,
                y: (leftEyeLeft.y + leftEyeRight.y) / 2
            };

            const rightEyeCenter = {
                x: (rightEyeLeft.x + rightEyeRight.x) / 2,
                y: (rightEyeLeft.y + rightEyeRight.y) / 2
            };

            const eyeCenter = {
                x: (leftEyeCenter.x + rightEyeCenter.x) / 2,
                y: (leftEyeCenter.y + rightEyeCenter.y) / 2
            };

            const noseTip = landmarks[1];
            const eyeToNoseDistance = noseTip ? (noseTip.y - eyeCenter.y) : 0.03;

            return {
                x: eyeCenter.x,
                y: eyeCenter.y + (eyeToNoseDistance * 0.6)
            };
        } catch (error) {
            return null;
        }
    }

    calculateFaceSize(landmarks) {
        if (!landmarks || landmarks.length === 0) return 0.2;

        try {
            const topPoint = landmarks[10];
            const bottomPoint = landmarks[152];
            const leftPoint = landmarks[234];
            const rightPoint = landmarks[454];

            if (!topPoint || !bottomPoint || !leftPoint || !rightPoint) {
                return 0.2;
            }

            const height = Math.abs(bottomPoint.y - topPoint.y);
            const width = Math.abs(rightPoint.x - leftPoint.x);
            return Math.max(height, width);
        } catch (error) {
            return 0.2;
        }
    }

    calculateHeadRotation(landmarks) {
        if (!landmarks || landmarks.length === 0) return 0;

        try {
            const leftEyeLeft = landmarks[33];
            const leftEyeRight = landmarks[133];
            const rightEyeLeft = landmarks[362];
            const rightEyeRight = landmarks[263];

            const leftEyeCenter = {
                x: (leftEyeLeft.x + leftEyeRight.x) / 2,
                y: (leftEyeLeft.y + leftEyeRight.y) / 2
            };

            const rightEyeCenter = {
                x: (rightEyeLeft.x + rightEyeRight.x) / 2,
                y: (rightEyeLeft.y + rightEyeRight.y) / 2
            };

            const deltaX = rightEyeCenter.x - leftEyeCenter.x;
            const deltaY = rightEyeCenter.y - leftEyeCenter.y;
            return Math.atan2(deltaY, deltaX) * (180 / Math.PI);
        } catch (error) {
            return 0;
        }
    }

    // Get current registry stats for debugging
    getStats() {
        return {
            trackedFaces: this.trackedFaces.size,
            faceIds: Array.from(this.trackedFaces.keys()),
            maskAssignments: Array.from(this.trackedFaces.values()).map(f => ({
                id: f.faceId,
                mask: f.maskNumber,
                missing: f.framesMissing
            }))
        };
    }

    // Reset/clear all tracked faces (useful for debugging or manual cleanup)
    reset() {
        console.log('🔄 Face registry reset - clearing all tracked faces');
        this.trackedFaces.clear();
        this.nextFaceId = 1;
    }
}

const PhotoBooth = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const captureAreaRef = useRef(null);
    const printAreaRef = useRef(null);

    const [isModelLoaded, setIsModelLoaded] = useState(false);
    const [capturedImage, setCapturedImage] = useState(null);
    const [firebaseUrl, setFirebaseUrl] = useState('');
    const [faces, setFaces] = useState([]);
    const animationFrameRef = useRef();
    const faceLandmarkerRef = useRef(null);

    // Smooth animation state for Instagram/Snapchat-like behavior
    const [smoothFaces, setSmoothFaces] = useState([]);
    const prevFaceData = useRef([]);

    // Face tracking registry for persistent mask assignment
    const faceRegistryRef = useRef(new FaceRegistry());

    // Photo booth flow states
    const [currentScreen, setCurrentScreen] = useState('camera'); // 'camera', 'countdown', 'loading', 'result'
    const [countdown, setCountdown] = useState(5);
    const [showShutter, setShowShutter] = useState(false);
    const [capturedVideoOnly, setCapturedVideoOnly] = useState(null); // Store video-only capture
    const [capturedMaskData, setCapturedMaskData] = useState([]); // Store mask positions for result

    // Utility functions for smooth animations with MediaPipe landmarks
    const lerp = (a, b, t) => a + (b - a) * t;

    const calculateHeadRotation = (landmarks) => {
        if (!landmarks || landmarks.length === 0) return 0;

        // MediaPipe FaceMesh landmarks - use key eye points
        // Left eye corners: 33 (left), 133 (right)
        // Right eye corners: 362 (left), 263 (right)
        const leftEyeLeft = landmarks[33];
        const leftEyeRight = landmarks[133];
        const rightEyeLeft = landmarks[362];
        const rightEyeRight = landmarks[263];

        // Calculate eye centers
        const leftEyeCenter = {
            x: (leftEyeLeft.x + leftEyeRight.x) / 2,
            y: (leftEyeLeft.y + leftEyeRight.y) / 2
        };

        const rightEyeCenter = {
            x: (rightEyeLeft.x + rightEyeRight.x) / 2,
            y: (rightEyeLeft.y + rightEyeRight.y) / 2
        };

        // Calculate rotation angle based on eye line
        const deltaX = rightEyeCenter.x - leftEyeCenter.x;
        const deltaY = rightEyeCenter.y - leftEyeCenter.y;
        const rotationAngle = Math.atan2(deltaY, deltaX) * (180 / Math.PI);

        return rotationAngle;
    };

    const calculateFaceCenter = (landmarks) => {
        if (!landmarks || landmarks.length === 0) return null;

        try {
            // Use key eye landmarks for accurate center calculation
            const leftEyeLeft = landmarks[33];
            const leftEyeRight = landmarks[133];
            const rightEyeLeft = landmarks[362];
            const rightEyeRight = landmarks[263];

            if (!leftEyeLeft || !leftEyeRight || !rightEyeLeft || !rightEyeRight) {
                console.log('Missing eye landmarks');
                return null;
            }

            // Calculate eye centers
            const leftEyeCenter = {
                x: (leftEyeLeft.x + leftEyeRight.x) / 2,
                y: (leftEyeLeft.y + leftEyeRight.y) / 2
            };

            const rightEyeCenter = {
                x: (rightEyeLeft.x + rightEyeRight.x) / 2,
                y: (rightEyeLeft.y + rightEyeRight.y) / 2
            };

            // Calculate center between eyes, then move it down towards nose area
            const eyeCenter = {
                x: (leftEyeCenter.x + rightEyeCenter.x) / 2,
                y: (leftEyeCenter.y + rightEyeCenter.y) / 2
            };

            // Get nose tip landmark (point 1) to help position mask better
            const noseTip = landmarks[1];

            // Move center down by 60% of the distance from eyes to nose
            const eyeToNoseDistance = noseTip ? (noseTip.y - eyeCenter.y) : 0.03; // fallback
            const center = {
                x: eyeCenter.x,
                y: eyeCenter.y + (eyeToNoseDistance * 0.6) // Move down 60% towards nose to avoid mouth
            };

            // console.log('Face center (normalized):', center); // Reduce logging
            return center;
        } catch (error) {
            console.error('Error calculating face center:', error);
            return null;
        }
    };

    const calculateFaceSize = (landmarks) => {
        if (!landmarks || landmarks.length === 0) return 200; // Default size

        try {
            // Use face oval landmarks to calculate size
            // Top of forehead: 10, Bottom of chin: 152
            // Left cheek: 234, Right cheek: 454
            const topPoint = landmarks[10];
            const bottomPoint = landmarks[152];
            const leftPoint = landmarks[234];
            const rightPoint = landmarks[454];

            if (!topPoint || !bottomPoint || !leftPoint || !rightPoint) {
                console.log('Missing face outline landmarks');
                return 200;
            }

            const height = Math.abs(bottomPoint.y - topPoint.y);
            const width = Math.abs(rightPoint.x - leftPoint.x);

            // Return size in normalized coordinates
            const size = Math.max(height, width);
            // console.log('Face size (normalized):', size); // Reduce logging
            return size;
        } catch (error) {
            console.error('Error calculating face size:', error);
            return 200;
        }
    };

    // Enhanced smooth interpolation with persistent face tracking
    const applySmoothInterpolation = useCallback((currentFaces) => {
        const smoothingFactor = 0.2; // Slightly more responsive for MediaPipe

        // Use face registry to get persistent mask assignments
        const trackedFaces = faceRegistryRef.current.processDetections(currentFaces);

        const newSmoothFaces = trackedFaces.map((faceData, index) => {
            const { landmarks, center: faceCenter, size: faceSize, rotation, maskNumber } = faceData;

            if (!faceCenter) return null;

            // Get canvas element dimensions (which displays the mirrored video)
            const canvasElement = canvasRef.current;
            const videoElement = videoRef.current;
            const canvasRect = canvasElement.getBoundingClientRect();

            // Calculate the actual displayed video dimensions considering object-fit: cover
            const videoAspectRatio = videoElement.videoWidth / videoElement.videoHeight;
            const displayAspectRatio = canvasRect.width / canvasRect.height;

            let displayedVideoWidth, displayedVideoHeight, offsetX, offsetY;

            if (videoAspectRatio > displayAspectRatio) {
                // Video is wider than display area, so height fills and width is cropped
                displayedVideoHeight = canvasRect.height;
                displayedVideoWidth = canvasRect.height * videoAspectRatio;
                offsetX = (displayedVideoWidth - canvasRect.width) / 2;
                offsetY = 0;
            } else {
                // Video is taller than display area, so width fills and height is cropped
                displayedVideoWidth = canvasRect.width;
                displayedVideoHeight = canvasRect.width / videoAspectRatio;
                offsetX = 0;
                offsetY = (displayedVideoHeight - canvasRect.height) / 2;
            }

            // Convert MediaPipe normalized coordinates to displayed video coordinates
            // Mirror the x-coordinate to match the canvas-mirrored video
            const mirroredX = 1 - faceCenter.x; // Flip x-coordinate for mirrored canvas display
            const pixelX = (mirroredX * displayedVideoWidth) - offsetX;
            const pixelY = (faceCenter.y * displayedVideoHeight) - offsetY;

            // Mirror the rotation angle to match the canvas-mirrored video
            const mirroredRotation = -rotation; // Invert rotation for mirrored canvas display

            // Scale mask size to match the displayed video scale
            const videoScale = displayedVideoWidth / videoElement.videoWidth;
            
            // Improved mask scaling: make mask size directly proportional to face size
            // faceSize is in normalized coordinates (0-1), so scale it to display dimensions
            const baseMaskSize = faceSize * displayedVideoHeight * 2.5; // 2.5x face height for good coverage
            const maskSize = Math.max(100, Math.min(800, baseMaskSize)); // Clamp size with better limits
            const scaledMaskSize = maskSize;

            // Debug only occasionally with registry info
            if (index === 0 && Math.random() < 0.02) { // Log 2% of frames
                const stats = faceRegistryRef.current.getStats();
                console.log('🎭 Face Tracking:', {
                    currentFaces: trackedFaces.length,
                    maskNumber,
                    faceId: faceData.faceId,
                    totalTracked: stats.trackedFaces,
                    activeMasks: stats.maskAssignments,
                    finalPos: `${Math.round(pixelX)}, ${Math.round(pixelY)}`
                });
            }

            // Get previous data for this specific face (using faceId for consistency)
            const prevData = prevFaceData.current.find(data => data.faceId === faceData.faceId);

            let smoothData;
            if (prevData) {
                // Apply smooth interpolation
                smoothData = {
                    x: lerp(prevData.x, pixelX, smoothingFactor),
                    y: lerp(prevData.y, pixelY, smoothingFactor),
                    rotation: lerp(prevData.rotation, mirroredRotation, smoothingFactor),
                    size: lerp(prevData.size, scaledMaskSize, smoothingFactor),
                    maskNumber,
                    faceId: faceData.faceId
                };
            } else {
                // First frame for this face - no previous data
                smoothData = {
                    x: pixelX,
                    y: pixelY,
                    rotation: mirroredRotation,
                    size: scaledMaskSize,
                    maskNumber,
                    faceId: faceData.faceId
                };
            }

            return smoothData;
        }).filter(face => face !== null);

        // Update prevFaceData to store by faceId for next frame
        prevFaceData.current = newSmoothFaces.map(face => ({
            faceId: face.faceId,
            x: face.x,
            y: face.y,
            rotation: face.rotation,
            size: face.size
        }));

        // Ensure we clear faces immediately when none are detected
        if (trackedFaces.length === 0) {
            prevFaceData.current = [];
        }

        setSmoothFaces(newSmoothFaces);
    }, []);

    // Initialize MediaPipe FaceLandmarker
    useEffect(() => {
        const initializeMediaPipe = async () => {
            console.log('Initializing MediaPipe FaceLandmarker...');
            try {
                // Initialize the vision tasks with correct path
                const vision = await FilesetResolver.forVisionTasks(
                    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
                );
                console.log('Vision tasks initialized');

                // Create FaceLandmarker instance
                const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                        delegate: "CPU" // Use CPU for better compatibility
                    },
                    outputFaceBlendshapes: false,
                    outputFacialTransformationMatrixes: false,
                    runningMode: "VIDEO",
                    numFaces: 5 // Support up to 5 faces for group photos
                });
                console.log('FaceLandmarker created');

                faceLandmarkerRef.current = faceLandmarker;
                console.log('MediaPipe FaceLandmarker initialized successfully');
                setIsModelLoaded(true);
            } catch (error) {
                console.error('Error initializing MediaPipe:', error);
            }
        };
        initializeMediaPipe();
    }, []);

    // Store the video stream to maintain it across screen changes
    const videoStreamRef = useRef(null);

    // Start video stream
    useEffect(() => {
        const startVideo = async () => {
            if (isModelLoaded) {
                console.log('Starting video stream...');
                try {
                    // Check if getUserMedia is available
                    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                        throw new Error('getUserMedia is not supported in this browser');
                    }

                    console.log('Requesting camera access...');
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: {
                            width: { ideal: 1920 },
                            height: { ideal: 1080 },
                            aspectRatio: { ideal: 16 / 9 },
                            facingMode: 'user' // Front-facing camera
                        }
                    });
                    console.log('Camera access granted, stream received:', stream);
                    
                    videoStreamRef.current = stream; // Store stream reference
                    if (videoRef.current) {
                        videoRef.current.srcObject = stream;
                        console.log('Video stream applied to video element');
                        
                        // Wait for video to be ready
                        videoRef.current.onloadedmetadata = () => {
                            console.log('Video metadata loaded, video dimensions:', 
                                videoRef.current.videoWidth, 'x', videoRef.current.videoHeight);
                            videoRef.current.play().then(() => {
                                console.log('Video playback started successfully');
                            }).catch(e => {
                                console.error('Video play failed:', e);
                            });
                        };
                        
                        console.log('Video stream started');
                    } else {
                        console.error('Video ref is null');
                    }
                } catch (error) {
                    console.error('Error accessing webcam:', error);
                    
                    // Provide more specific error messages
                    if (error.name === 'NotAllowedError') {
                        alert('Camera access denied. Please allow camera permissions and refresh the page.');
                    } else if (error.name === 'NotFoundError') {
                        alert('No camera found. Please make sure a camera is connected.');
                    } else if (error.name === 'NotReadableError') {
                        alert('Camera is being used by another application. Please close other camera apps and try again.');
                    } else {
                        alert(`Camera error: ${error.message}`);
                    }
                }
            }
        };
        startVideo();
    }, [isModelLoaded]);

    // Ensure video stream is applied when screen changes or video element changes
    useEffect(() => {
        if (videoStreamRef.current && videoRef.current) {
            console.log(`Screen changed to ${currentScreen} - ensuring video stream`);

            // Always reapply stream on screen change to be safe
            videoRef.current.srcObject = videoStreamRef.current;

            // Force play the video
            setTimeout(() => {
                if (videoRef.current) {
                    videoRef.current.play().catch(e => console.log('Video play failed:', e));
                }
            }, 100); // Small delay to ensure element is ready
        }
    }, [currentScreen]); // Reapply when screen changes

    // Draw mirrored video on canvas
    const drawMirroredVideo = useCallback(() => {
        if (videoRef.current && canvasRef.current) {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');

            if (video.videoWidth > 0 && video.videoHeight > 0) {
                // Set canvas dimensions to match video
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                // Clear canvas
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Save context
                ctx.save();

                // Mirror the canvas horizontally
                ctx.scale(-1, 1);
                ctx.translate(-canvas.width, 0);

                // Draw the video frame
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                // Restore context
                ctx.restore();
            }
        }
    }, []);

    // MediaPipe face detection with high-performance real-time tracking
    const detectFaces = useCallback(async () => {
        if (videoRef.current && faceLandmarkerRef.current && isModelLoaded) {
            const video = videoRef.current;

            if (video.videoWidth > 0 && video.videoHeight > 0) {
                try {
                    // Draw mirrored video on canvas
                    drawMirroredVideo();

                    const startTimeMs = performance.now();

                    // Detect faces using MediaPipe
                    const results = faceLandmarkerRef.current.detectForVideo(video, startTimeMs);

                    // Extract landmarks from results
                    const faceLandmarks = results.faceLandmarks || [];

                    // Log detection results occasionally (every 30 frames)
                    if (startTimeMs % 1000 < 33) {
                        console.log('Group detection:', {
                            facesDetected: faceLandmarks.length,
                            maxFaces: 5,
                            landmarksPerFace: faceLandmarks[0]?.length || 0
                        });
                    }

                    setFaces(faceLandmarks);

                    // Apply smooth interpolation for Instagram-like filter behavior
                    if (faceLandmarks.length > 0) {
                        applySmoothInterpolation(faceLandmarks);
                    } else {
                        setSmoothFaces([]);
                    }
                } catch (error) {
                    console.error('Error detecting faces with MediaPipe:', error);
                }
            }
        }

        // Continue the animation loop - 30fps for responsive mask cleanup  
        setTimeout(() => {
            animationFrameRef.current = requestAnimationFrame(detectFaces);
        }, 33); // ~30fps for better responsiveness
    }, [isModelLoaded, applySmoothInterpolation, drawMirroredVideo]);

    // Start face detection loop
    useEffect(() => {
        if (isModelLoaded) {
            detectFaces();
        }

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
            // Clean up face registry when component unmounts or detection stops
            faceRegistryRef.current.reset();
        };
    }, [detectFaces, isModelLoaded]);



    // Start photo capture with countdown
    const startPhotoCapture = () => {
        setCurrentScreen('countdown');
        setCountdown(5);
        startCountdown();
    };

    // Countdown function
    const startCountdown = () => {
        const countdownInterval = setInterval(() => {
            setCountdown((prev) => {
                if (prev <= 1) {
                    clearInterval(countdownInterval);
                    // Trigger shutter effect and capture
                    setTimeout(() => {
                        setShowShutter(true);
                        capturePhoto();
                        setTimeout(() => setShowShutter(false), 200);
                    }, 500);
                    return 0;
                }
                return prev - 1;
            });
        }, 1000);
    };

    // Capture screenshot - simple method that works, then flip video in result
    const capturePhoto = async () => {
        setCurrentScreen('loading');

        try {
            console.log('📸 Starting simple capture...');

            if (!captureAreaRef.current) {
                console.error('❌ Capture area ref is null!');
                throw new Error('Capture area not found');
            }

            // Simple working capture - just like before
            const canvas = await html2canvas(captureAreaRef.current, {
                useCORS: true,
                allowTaint: true,
                scale: 1,
                logging: false
            });

            const imageData = canvas.toDataURL('image/png');
            setCapturedImage(imageData);

            // Also set as video-only for the new result display
            setCapturedVideoOnly(imageData);

            // Store current mask data for overlay in result (empty for now since we capture everything)
            setCapturedMaskData([]);

            console.log('📸 Simple capture successful!');

            // Upload to Firebase (but don't let it block the UI)
            uploadToFirebase(imageData).catch(error => {
                console.warn('Firebase upload failed, but continuing with local image:', error);
            });

            // Always show result screen after capture, regardless of upload status
            setTimeout(() => {
                setCurrentScreen('result');
            }, 1500); // Give Firebase a chance to upload, but don't wait indefinitely

        } catch (error) {
            console.error('❌ Capture failed:', error);
            alert('Failed to capture photo. Please try again.');
            setCurrentScreen('camera'); // Return to camera on error
        }
    };

    // Reset to camera screen
    const takeNewPhoto = () => {
        setCapturedImage(null);
        setCapturedVideoOnly(null);
        setCapturedMaskData([]);
        setFirebaseUrl('');
        // Clear any stuck masks when returning to camera
        faceRegistryRef.current.reset();
        setSmoothFaces([]);
        setCurrentScreen('camera');
    };

    // Upload to Firebase Storage
    const uploadToFirebase = async (imageData) => {
        console.log('Uploading to Firebase...');
        try {
            const timestamp = Date.now();
            const fileName = `photo-${timestamp}.png`;
            const storageRef = ref(storage, `photos/${fileName}`);

            // Remove data:image/png;base64, prefix
            const base64Data = imageData.split(',')[1];

            await uploadString(storageRef, base64Data, 'base64');
            const downloadURL = await getDownloadURL(storageRef);

            setFirebaseUrl(downloadURL);
            console.log('Upload successful:', downloadURL);
        } catch (error) {
            console.error('Error uploading to Firebase:', error);
            // For demo purposes, set a placeholder URL
            setFirebaseUrl('https://placeholder-image-url.com/photo.png');
        }
    };



    // Render different screens based on current state
    const renderScreen = () => {
        switch (currentScreen) {
            case 'camera':
                return renderCameraScreen();
            case 'countdown':
                return renderCountdownScreen();
            case 'loading':
                return renderLoadingScreen();
            case 'result':
                return renderResultScreen();
            default:
                return renderCameraScreen();
        }
    };

    // Camera screen with video stream
    const renderCameraScreen = () => (
        <div style={{
            minHeight: '100vh',
            background: 'linear-gradient(135deg, #FFE4F0 0%, #FFB8E1 30%, #EE9ABF 70%, #D687A3 100%)',
            padding: '20px',
            fontFamily: 'Nunito, sans-serif'
        }}>
            <div style={{
                textAlign: 'center',
                marginBottom: '20px',
                position: 'relative'
            }}>
                <h1 style={{
                    fontSize: 'clamp(2rem, 5vw, 3.5rem)',
                    fontWeight: '800',
                    color: '#FFFFFF',
                    textShadow: '3px 3px 0px #D687A3, 6px 6px 20px rgba(214, 135, 163, 0.4)',
                    margin: '0',

                    letterSpacing: '2px'
                }}>
                    Moca Photo Booth
                </h1>

            </div>

            {/* Camera and Capture Area - 16:9 Responsive */}
            <div
                ref={captureAreaRef}
                style={{
                    position: 'relative',
                    width: 'min(95vw, calc(80vh * 16/9))', // Constrain by viewport
                    aspectRatio: '16/9',
                    margin: '0 auto 20px',
                    border: '6px solid #FFFFFF',
                    borderRadius: '25px',
                    overflow: 'hidden',
                    backgroundColor: '#000',
                    boxShadow: '0 8px 30px rgba(214, 135, 163, 0.4), 0 0 0 3px #EE9ABF, 0 0 20px rgba(238, 154, 191, 0.3)'
                }}
            >
                {/* Hidden video element for MediaPipe processing */}
                <video
                    ref={videoRef}
                    autoPlay
                    muted
                    style={{
                        display: 'none' // Hide the original video element
                    }}
                    onLoadedMetadata={() => console.log('Video metadata loaded - 16:9 stream')}
                />

                {/* Canvas for mirrored video display */}
                <canvas
                    ref={canvasRef}
                    style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                        display: 'block'
                    }}
                />



                {/* Mask Overlays - Persistent tracking with consistent assignments */}
                {smoothFaces.map((faceData, index) => {
                    // Use persistent mask number from face tracking registry
                    const maskNumber = faceData.maskNumber;
                    const maskSrc = `/mask/mask${maskNumber}.png`;

                    return (
                        <img
                            key={faceData.faceId} // Use persistent faceId as key for React rendering
                            src={maskSrc}
                            alt={`Face Mask ${maskNumber}`}
                            style={{
                                position: 'absolute',
                                left: `${faceData.x}px`,
                                top: `${faceData.y}px`,
                                width: `${faceData.size}px`,
                                height: `${faceData.size}px`,
                                transform: `translate(-50%, -50%) rotate(${faceData.rotation}deg)`,
                                pointerEvents: 'none',
                                zIndex: 50,
                                transition: 'none', // Disable CSS transitions - we handle smoothing manually
                                opacity: 0.95 // Slight transparency for better layering
                            }}
                            onError={(e) => {
                                console.log(`Mask ${maskNumber} not found`);
                                e.target.style.display = 'none';
                            }}
                        />
                    );
                })}

                {/* Frame Overlay */}
                <img
                    src="/frame.png"
                    alt="Decorative Frame"
                    style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        pointerEvents: 'none',
                        zIndex: 100
                    }}
                    onError={(e) => {
                        console.log('Frame not found');
                        e.target.style.display = 'none';
                    }}
                />

            </div>



            {/* Controls */}
            <div style={{
                textAlign: 'center',
                marginBottom: '20px',
                padding: '20px'
            }}>
                <button
                    onClick={startPhotoCapture}
                    disabled={!isModelLoaded}
                    style={{
                        padding: '18px 40px',
                        fontSize: 'clamp(16px, 4vw, 24px)',
                        fontWeight: '700',
                        background: isModelLoaded ? 
                            'linear-gradient(45deg, #EE9ABF, #FFB8E1, #EE9ABF)' : 
                            'linear-gradient(45deg, #D3D3D3, #E8E8E8)',
                        color: '#FFFFFF',
                        border: '4px solid #FFFFFF',
                        borderRadius: '50px',
                        cursor: isModelLoaded ? 'pointer' : 'not-allowed',
                        opacity: !isModelLoaded ? 0.6 : 1,
                        minWidth: '200px',
                        boxShadow: isModelLoaded ? 
                            '0 8px 25px rgba(238, 154, 191, 0.4), 0 0 0 2px #EE9ABF' : 
                            '0 4px 15px rgba(0,0,0,0.2)',
                        textShadow: '2px 2px 4px rgba(214, 135, 163, 0.8)',
                        transition: 'all 0.3s ease',
                        transform: 'scale(1)'
                    }}
                    onMouseOver={(e) => {
                        if (isModelLoaded) {
                            e.target.style.transform = 'scale(1.05)';
                        }
                    }}
                    onMouseOut={(e) => {
                        if (isModelLoaded) {
                            e.target.style.transform = 'scale(1)';
                        }
                    }}
                >
                    📸 SNAP PHOTO
                </button>
            </div>
        </div>
    );

    // Countdown screen with live video stream
    const renderCountdownScreen = () => (
        <div style={{
            fontFamily: 'Nunito, sans-serif',
            minHeight: '100vh',
            background: 'linear-gradient(135deg, #FFE4F0 0%, #FFB8E1 30%, #EE9ABF 70%, #D687A3 100%)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '20px'
        }}>


            {/* Camera and Capture Area - 16:9 Responsive with Live Stream */}
            <div
                ref={captureAreaRef}
                style={{
                    position: 'relative',
                    width: 'min(95vw, calc(80vh * 16/9))', // Constrain by viewport
                    aspectRatio: '16/9',
                    margin: '0 auto 0',
                    border: '6px solid #FFFFFF',
                    borderRadius: '25px',
                    overflow: 'hidden',
                    backgroundColor: '#000', // Match camera screen
                    boxShadow: '0 8px 30px rgba(214, 135, 163, 0.4), 0 0 0 3px #EE9ABF, 0 0 20px rgba(238, 154, 191, 0.3)'
                }}
            >
                {/* Hidden video element for MediaPipe processing */}
                <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    style={{
                        display: 'none' // Hide the original video element
                    }}
                    onLoadedMetadata={() => {
                        console.log('Countdown video metadata loaded - ensuring stream');
                        if (videoStreamRef.current && videoRef.current && !videoRef.current.srcObject) {
                            videoRef.current.srcObject = videoStreamRef.current;
                            videoRef.current.play().catch(e => console.log('Video play error:', e));
                        }
                    }}
                    onCanPlay={() => {
                        console.log('Countdown video can play - starting playback');
                        if (videoRef.current) {
                            videoRef.current.play().catch(e => console.log('Video play error:', e));
                        }
                    }}
                />

                {/* Canvas for mirrored video display */}
                <canvas
                    ref={canvasRef}
                    style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                        display: 'block'
                    }}
                />

                {/* Mask Overlays - Persistent tracking with consistent assignments */}
                {smoothFaces.map((faceData, index) => {
                    // Use persistent mask number from face tracking registry
                    const maskNumber = faceData.maskNumber;
                    const maskSrc = `/mask/mask${maskNumber}.png`;

                    return (
                        <img
                            key={faceData.faceId} // Use persistent faceId as key for React rendering
                            src={maskSrc}
                            alt={`Face Mask ${maskNumber}`}
                            style={{
                                position: 'absolute',
                                left: `${faceData.x}px`,
                                top: `${faceData.y}px`,
                                width: `${faceData.size}px`,
                                height: `${faceData.size}px`,
                                transform: `translate(-50%, -50%) rotate(${faceData.rotation}deg)`,
                                pointerEvents: 'none',
                                zIndex: 50,
                                transition: 'none', // Disable CSS transitions - we handle smoothing manually
                                opacity: 0.95 // Slight transparency for better layering
                            }}
                            onError={(e) => {
                                console.log(`Mask ${maskNumber} not found`);
                                e.target.style.display = 'none';
                            }}
                        />
                    );
                })}

                {/* Frame Overlay */}
                <img
                    src="/frame.png"
                    alt="Decorative Frame"
                    style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        pointerEvents: 'none',
                        zIndex: 100
                    }}
                    onError={(e) => {
                        console.log('Frame not found');
                        e.target.style.display = 'none';
                    }}
                />

                {/* Countdown Numbers - Kawaii Style */}
                <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    color: '#FFFFFF',
                    zIndex: 100,
                    textAlign: 'center'
                }}>
                    {countdown > 0 ? (
                        <div style={{
                            fontSize: 'clamp(8rem, 20vw, 16rem)',
                            fontWeight: '900',
                            fontFamily: 'Nunito, sans-serif',
                            textShadow: '4px 4px 0px #D687A3, 8px 8px 20px rgba(214, 135, 163, 0.6), 0 0 40px rgba(238, 154, 191, 0.8)',
                            color: '#FFFFFF'
                        }}>
                            {countdown}
                        </div>
                    ) : null}
                </div>

                {/* Shutter effect */}
                {showShutter && (
                    <div style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        backgroundColor: 'white',
                        zIndex: 200,
                        animation: 'flash 0.2s ease-out'
                    }} />
                )}

            </div>

            {/* Status Overlay - OUTSIDE capture area */}
            <div style={{
                background: 'rgba(0,0,0,0.7)',
                color: 'white',
                padding: '5px 10px',
                borderRadius: '5px',
                fontSize: '12px',
                marginTop: '5px',
                textAlign: 'center',
                width: 'min(95vw, calc(80vh * 16/9))',
                margin: '5px auto 0'
            }}>
                {!isModelLoaded ? 'Loading MediaPipe...' :
                    smoothFaces.length === 0 ? 'No faces detected' :
                        smoothFaces.length === 1 ? `1 person tracked (mask${smoothFaces[0].maskNumber})` :
                            `${smoothFaces.length} people tracked (masks: ${smoothFaces.map(f => f.maskNumber).join(', ')})`}
            </div>
        </div>
    );

    // Loading screen
    const renderLoadingScreen = () => (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100vh',
            background: 'linear-gradient(135deg, #FFE4F0 0%, #FFB8E1 30%, #EE9ABF 70%, #D687A3 100%)',
            color: '#FFFFFF',
            fontFamily: 'Nunito, sans-serif',
            position: 'relative',
            overflow: 'hidden'
        }}>

            <div style={{
                fontSize: '6rem',
                marginBottom: '2rem',
                animation: 'bounce 1.5s infinite',
                textShadow: '4px 4px 0px #D687A3'
            }}>
                📸 ✨ 🌸
            </div>
            <h1 style={{ 
                fontSize: 'clamp(1.8rem, 4vw, 3rem)', 
                marginBottom: '2rem',
                fontWeight: '800',
                textAlign: 'center',
                textShadow: '3px 3px 0px #D687A3',
                letterSpacing: '2px'
            }}>
                🎀 Creating Your Magical Moment! 🎀
            </h1>
            <div style={{
                width: '300px',
                height: '12px',
                backgroundColor: 'rgba(255,255,255,0.4)',
                borderRadius: '20px',
                overflow: 'hidden',
                border: '3px solid #FFFFFF',
                boxShadow: '0 4px 15px rgba(238, 154, 191, 0.3)'
            }}>
                <div style={{
                    width: '100%',
                    height: '100%',
                    background: 'linear-gradient(90deg, #EE9ABF, #FFB8E1, #EE9ABF)',
                    animation: 'loading 2s ease-in-out infinite',
                    borderRadius: '15px'
                }} />
            </div>
            <p style={{
                marginTop: '1.5rem',
                fontSize: '1.2rem',
                fontWeight: '600',
                textShadow: '2px 2px 0px #D687A3'
            }}>
                ✨ Adding kawaii sparkles... ✨
            </p>
        </div>
    );

    // Result screen
    const renderResultScreen = () => (
        <div style={{
            padding: '30px',
            textAlign: 'center',
            background: 'linear-gradient(135deg, #FFE4F0 0%, #FFB8E1 30%, #EE9ABF 70%, #D687A3 100%)',
            minHeight: '100vh',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            fontFamily: 'Nunito, sans-serif',
            position: 'relative',
            overflow: 'hidden'
        }}>

            <h1 style={{ 
                fontSize: 'clamp(2rem, 5vw, 4rem)', 
                marginBottom: '3rem', 
                color: '#FFFFFF',
                fontWeight: '900',
                textShadow: '4px 4px 0px #D687A3, 8px 8px 25px rgba(214, 135, 163, 0.6)',
                letterSpacing: '3px'
            }}>
                ✨ Your Magical Photo is Ready! ✨
            </h1>

            {capturedVideoOnly && (
                <div style={{
                    display: 'flex',
                    flexDirection: 'row',
                    alignItems: 'flex-start',
                    justifyContent: 'center',
                    gap: '3rem',
                    flexWrap: 'wrap', // Allow wrapping on small screens
                    maxWidth: '1200px',
                    margin: '0 auto'
                }}>
                    {/* QR Code on the left */}
                    {firebaseUrl && (
                        <div style={{
                            background: 'linear-gradient(45deg, #FFFFFF, #FFF5FB)',
                            padding: '3rem',
                            borderRadius: '30px',
                            boxShadow: '0 10px 35px rgba(238, 154, 191, 0.3), 0 0 0 4px #EE9ABF',
                            flex: '0 0 auto', // Don't grow/shrink
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            minWidth: '320px',
                            border: '6px solid #FFFFFF'
                        }}>
                            <h3 style={{ 
                                marginBottom: '1.5rem', 
                                color: '#EE9ABF',
                                fontSize: '1.5rem',
                                fontWeight: '800',
                                textShadow: '2px 2px 4px rgba(238, 154, 191, 0.3)',
                                letterSpacing: '1px'
                            }}>
                                📱 Scan to Download
                            </h3>
                            <div style={{
                                padding: '15px',
                                background: 'linear-gradient(45deg, #EE9ABF, #FFB8E1)',
                                borderRadius: '20px',
                                border: '4px solid #FFFFFF',
                                boxShadow: '0 6px 20px rgba(238, 154, 191, 0.4)'
                            }}>
                                <QRCodeCanvas value={firebaseUrl} size={270} />
                            </div>
                            <p style={{ 
                                marginTop: '1.5rem', 
                                color: '#D687A3',
                                textAlign: 'center',
                                fontSize: '1.1rem',
                                fontWeight: '700',
                                lineHeight: '1.4'
                            }}>
                                ✨ Scan with your phone to save your magical memory! ✨
                            </p>
                        </div>
                    )}

                    {/* Photo and Button on the right */}
                    <div style={{
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        gap: '2rem',
                        flex: '1 1 300px',
                        alignSelf: 'flex-start'
                    }}>
                        {/* Display captured result (already mirrored from live video) */}
                        <img
                            src={capturedVideoOnly}
                            alt="Captured Moment"
                            style={{
                                maxWidth: '450px',
                                width: '100%',
                                height: 'auto',
                                border: '8px solid #FFFFFF',
                                borderRadius: '25px',
                                boxShadow: '0 15px 40px rgba(238, 154, 191, 0.4), 0 0 0 4px #EE9ABF, 0 0 25px rgba(238, 154, 191, 0.3)',
                                display: 'block',
                                margin: '0',
                                padding: '0',
                                verticalAlign: 'top'
                            }}
                        />

                        <button
                            onClick={takeNewPhoto}
                            style={{
                                padding: '20px 50px',
                                fontSize: 'clamp(1.2rem, 3vw, 2rem)',
                                fontWeight: '800',
                                background: 'linear-gradient(45deg, #EE9ABF, #FFB8E1, #EE9ABF)',
                                color: '#FFFFFF',
                                border: '6px solid #FFFFFF',
                                borderRadius: '50px',
                                cursor: 'pointer',
                                boxShadow: '0 10px 30px rgba(238, 154, 191, 0.4), 0 0 0 3px #EE9ABF',
                                transition: 'all 0.3s ease',
                                textShadow: '2px 2px 4px rgba(214, 135, 163, 0.8)',
                                letterSpacing: '2px',
                                minWidth: '280px',
                                fontFamily: 'Nunito, sans-serif'
                            }}
                            onMouseOver={(e) => {
                                e.target.style.transform = 'scale(1.08)';
                            }}
                            onMouseOut={(e) => {
                                e.target.style.transform = 'scale(1)';
                            }}
                        >
                            New Photo
                        </button>
                    </div>
                </div>
            )}

            {/* Print Layout (hidden on screen, visible when printing) */}
            <div
                ref={printAreaRef}
                style={{
                    display: 'none'
                }}
                className="print-only"
            >
                {capturedImage && (
                    <div style={{
                        width: '100%',
                        textAlign: 'center',
                        padding: '20px'
                    }}>
                        <h2>Photo Booth Memory</h2>
                        <img
                            src={capturedImage}
                            alt="Printed Memory"
                            style={{
                                maxWidth: '600px',
                                height: 'auto',
                                marginBottom: '20px'
                            }}
                        />
                        {firebaseUrl && (
                            <div>
                                <QRCodeCanvas value={firebaseUrl} size={100} />
                                <p>Scan to download digital copy</p>
                            </div>
                        )}
                        <p style={{ fontSize: '12px', marginTop: '20px' }}>
                            Created with Photo Booth App - {new Date().toLocaleDateString()}
                        </p>
                    </div>
                )}
            </div>
        </div>
    );

    return (
        <div style={{
            fontFamily: 'Arial, sans-serif',
            minHeight: '100vh',
            display: 'flex',
            flexDirection: 'column'
        }}>
            {renderScreen()}
        </div>
    );
};

export default PhotoBooth; 