package com.aarya.agricnxedge.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext

@Composable
fun AgriCNXEdgeTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    // We set dynamicColor to false by default so our custom Sage Green is forced to show,
    // overriding the user's Android system wallpaper colors.
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }

        darkTheme -> AgriDarkColorScheme
        else -> AgriLightColorScheme
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = AgriTypography, // Linked to our custom Type.kt
        content = content
    )
}