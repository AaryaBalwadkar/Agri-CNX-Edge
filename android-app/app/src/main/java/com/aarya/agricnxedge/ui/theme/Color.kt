package com.aarya.agricnxedge.ui.theme

import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.ui.graphics.Color

// ─────────────────────────────────────────────
//  Agri-CNX-Edge · "Earthy Precision" Palette
// ─────────────────────────────────────────────

// ── Primary · Sage Green ──────────────────────
val SageGreen          = Color(0xFF818C78)
val SageGreenLight     = Color(0xFFB1BD9F)   // tonal / on-primary-container
val SageGreenDark      = Color(0xFF535E4C)   // pressed / dark variant
val OnPrimaryWhite     = Color(0xFFFFFFFF)
val PrimaryContainer   = Color(0xFFD7E8C8)
val OnPrimaryContainer = Color(0xFF1B2716)

// ── Secondary · Forest Green ─────────────────
val ForestGreen          = Color(0xFF3E4437)
val ForestGreenLight     = Color(0xFF6B7263)
val ForestGreenDark      = Color(0xFF1E2219)
val OnSecondaryWhite     = Color(0xFFFFFFFF)
val SecondaryContainer   = Color(0xFFC4CCBA)
val OnSecondaryContainer = Color(0xFF0E1409)

// ── Tertiary · Harvest Amber (accent) ────────
val HarvestAmber         = Color(0xFFB5820A)
val HarvestAmberLight    = Color(0xFFE8B84B)
val TertiaryContainer    = Color(0xFFFFDEA0)
val OnTertiaryContainer  = Color(0xFF261900)

// ── Neutral / Background / Surface ───────────
/** High-contrast Bone White – reduces glare under direct sunlight in the field. */
val BoneWhite            = Color(0xFFF5F5F0)
val SurfaceVariant       = Color(0xFFE8E8E0)
val OutlineColor         = Color(0xFF8B8F84)
val OutlineVariant       = Color(0xFFC3C7BB)

// ── Camera Preview Placeholder ───────────────
val CameraPreviewBg      = Color(0xFF1A1C18)   // near-black to simulate viewfinder

// ── Semantic ─────────────────────────────────
val HealthyGreen  = Color(0xFF4CAF50)
val WarningAmber  = Color(0xFFFFC107)
val CriticalRed   = Color(0xFFD32F2F)
val InfoBlue      = Color(0xFF1976D2)

// ── Error ─────────────────────────────────────
val ErrorRed      = Color(0xFFBA1A1A)
val ErrorContainer = Color(0xFFFFDAD6)

// ─────────────────────────────────────────────
//  Material 3 Color Schemes
// ─────────────────────────────────────────────

val AgriLightColorScheme = lightColorScheme(
    // Primary
    primary            = SageGreen,
    onPrimary          = OnPrimaryWhite,
    primaryContainer   = PrimaryContainer,
    onPrimaryContainer = OnPrimaryContainer,

    // Secondary
    secondary            = ForestGreen,
    onSecondary          = OnSecondaryWhite,
    secondaryContainer   = SecondaryContainer,
    onSecondaryContainer = OnSecondaryContainer,

    // Tertiary
    tertiary            = HarvestAmber,
    tertiaryContainer   = TertiaryContainer,
    onTertiaryContainer = OnTertiaryContainer,

    // Backgrounds & Surfaces
    background      = BoneWhite,
    onBackground    = ForestGreenDark,
    surface         = BoneWhite,
    onSurface       = ForestGreenDark,
    surfaceVariant  = SurfaceVariant,
    onSurfaceVariant = ForestGreenLight,
    outline         = OutlineColor,
    outlineVariant  = OutlineVariant,

    // Error
    error            = ErrorRed,
    errorContainer   = ErrorContainer,
    onError          = OnPrimaryWhite,
    onErrorContainer = Color(0xFF410002),
)

val AgriDarkColorScheme = darkColorScheme(
    primary            = SageGreenLight,
    onPrimary          = SageGreenDark,
    primaryContainer   = SageGreenDark,
    onPrimaryContainer = SageGreenLight,

    secondary            = SecondaryContainer,
    onSecondary          = ForestGreenDark,
    secondaryContainer   = ForestGreenDark,
    onSecondaryContainer = SecondaryContainer,

    tertiary            = HarvestAmberLight,
    tertiaryContainer   = Color(0xFF3C2800),
    onTertiaryContainer = HarvestAmberLight,

    background      = Color(0xFF1A1C18),
    onBackground    = Color(0xFFE2E3D9),
    surface         = Color(0xFF1A1C18),
    onSurface       = Color(0xFFE2E3D9),
    surfaceVariant  = Color(0xFF424739),
    onSurfaceVariant = OutlineVariant,
    outline         = ForestGreenLight,

    error            = Color(0xFFFFB4AB),
    errorContainer   = Color(0xFF93000A),
)