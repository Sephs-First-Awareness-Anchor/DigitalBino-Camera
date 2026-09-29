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
        versionName = "0.1.0-probe"
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
