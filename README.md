# 📸 React Photo Booth App

A complete React photo booth application with face detection, mask overlays, Firebase storage, QR codes, and print functionality.

## ✨ Features

### 🎥 **Camera + Face Detection**
- Live webcam feed (640x480)
- Real-time face detection using `face-api.js`
- Multiple face detection support
- Eye landmark detection for precise mask positioning

### 😈 **Mask Overlays**
- 5 different mask designs (mask1.svg to mask5.svg)
- Automatic mask assignment (first face = mask1, second = mask2, etc.)
- Masks scale based on face width and position between eyes
- Colorful, fun mask designs with transparency

### 📸 **Screenshot Capture**
- Capture entire video area including overlays using `html2canvas`
- High-quality PNG screenshots
- Includes all visual elements (video, masks, frame)

### ☁️ **Firebase Storage**
- Automatic upload to Firebase Storage
- No authentication required (configurable)
- Files stored as `/photos/photo-TIMESTAMP.png`
- Public download URLs generated

### 🔳 **QR Code Generation**
- Dynamic QR codes using `qrcode.react`
- Contains Firebase download URL
- Scannable for instant photo sharing

### 🖼️ **Print with Custom Frame**
- Decorative golden frame overlay
- Professional print layout
- Includes QR code on printed version
- Custom print styles with `@media print`

### 📱 **Mobile-Friendly Design**
- Responsive layout
- Touch-friendly buttons
- Optimized for mobile devices

## 🚀 Setup Instructions

### 1. Install Dependencies
```bash
npm install
```

### 2. Download Face-API Models
The required models are already included in `public/models/`:
- `tiny_face_detector_model-*`
- `face_landmark_68_model-*`

### 3. Configure Firebase (Optional)
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create a new project
3. Enable Storage
4. Copy your config from Project Settings
5. Update the Firebase config in `src/PhotoBooth.js`

**Example Firebase Storage Rules (for demo):**
```javascript
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /{allPaths=**} {
      allow read, write: if true;
    }
  }
}
```

### 4. Start Development Server
```bash
npm start
```

Visit `http://localhost:3000` and allow camera access when prompted.

## 🎨 Customization

### Adding Custom Masks
1. Create SVG files in `public/mask/`
2. Name them `mask1.svg`, `mask2.svg`, etc.
3. Recommended size: 120x100px
4. Use semi-transparent colors for best effect

### Customizing the Frame
Edit `public/frame.svg` to create your own frame design:
- Maintain 640x480 dimensions
- Use transparent backgrounds
- Add your logo or event branding

### Styling
- Edit `src/App.css` for global styles
- Modify inline styles in `PhotoBooth.js` for component-specific styling
- Print styles are in the `@media print` section

## 🛠️ Technical Details

### Dependencies
- **React** - UI framework
- **face-api.js** - Face detection and landmarks
- **html2canvas** - Screenshot capture
- **Firebase** - Cloud storage
- **qrcode.react** - QR code generation

### Face Detection Process
1. Load TinyFaceDetector and FaceLandmark68Net models
2. Continuous detection loop (every 100ms)
3. Extract eye landmarks (positions 36-47)
4. Calculate center point between eyes
5. Position and scale masks based on eye distance

### File Structure
```
src/
├── PhotoBooth.js      # Main component
├── App.js            # App wrapper
├── App.css           # Styles + print CSS
└── index.js          # Entry point

public/
├── models/           # Face-API model files
├── mask/            # Mask SVG files
└── frame.svg        # Photo frame overlay
```

## 📝 Usage

1. **Allow Camera Access** - Browser will request webcam permission
2. **Position Faces** - Move into camera view for face detection
3. **Watch Masks Appear** - Masks automatically overlay on detected faces
4. **Take Photo** - Click "Take Photo" when ready
5. **Print or Share** - Use "Print Photo" or scan QR code

## 🎯 Browser Compatibility

- **Chrome** ✅ Full support
- **Firefox** ✅ Full support  
- **Safari** ✅ Full support
- **Edge** ✅ Full support
- **Mobile Safari** ✅ Full support
- **Chrome Mobile** ✅ Full support

## 📱 Mobile Support

The app is fully responsive and works great on mobile devices:
- Touch-friendly interface
- Responsive camera view
- Mobile-optimized buttons
- Proper aspect ratios

## 🖨️ Printing

The app includes special print styles:
- Only shows the captured photo and QR code when printing
- Removes navigation and controls
- Optimizes layout for standard paper sizes
- Includes event branding and date

## ⚠️ Notes

- **Camera Permission**: Required for webcam access
- **HTTPS**: Some browsers require HTTPS for camera access
- **Firebase**: Configure your own Firebase project for cloud storage
- **Models**: Face detection models are loaded from `/public/models/`
- **Performance**: Face detection runs every 100ms for smooth experience

## 🎉 Demo Features

Even without Firebase setup, the app will:
- Detect faces and show masks
- Capture screenshots
- Generate QR codes (with placeholder URL)
- Support printing

## 📄 License

MIT License - feel free to use for your events and projects!

---

Built with ❤️ using React and face-api.js
