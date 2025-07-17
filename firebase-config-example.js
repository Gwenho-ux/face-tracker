// Copy this file to src/firebase-config.js and replace with your actual Firebase config
// Get these values from your Firebase project settings

export const firebaseConfig = {
    apiKey: "your-api-key-here",
    authDomain: "your-project.firebaseapp.com",
    projectId: "your-project-id",
    storageBucket: "your-project.appspot.com",
    messagingSenderId: "123456789",
    appId: "your-app-id"
};

// Instructions:
// 1. Go to https://console.firebase.google.com/
// 2. Create a new project or select existing one
// 3. Go to Project Settings > General > Your apps
// 4. Add a web app if you haven't already
// 5. Copy the config object values above
// 6. Enable Firebase Storage in Storage section
// 7. Set up Storage rules (for demo, you can use public rules)

/* Example Storage Rules for demo (NOT secure for production):
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /{allPaths=**} {
      allow read, write: if true;
    }
  }
}
*/ 