// Authored by Sunni (Sir) Morningstar and Cael Devo
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.morningstar.stereoprobe"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.morningstar.stereoprobe"
        // getConcurrentCameraIds() / isConcurrentSessionConfigurationSupported() are API 30.
        minSdk = 30
        targetSdk = 34
        versionCode = 1
        versionName = "0.2.0-sweep"
        // OpenCV ships native libraries per ABI; the A16 is arm64.
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // OpenCV 4.9.0 official Android SDK (Java API + native libs) from Maven Central.
    implementation("org.opencv:opencv:4.9.0")
}
