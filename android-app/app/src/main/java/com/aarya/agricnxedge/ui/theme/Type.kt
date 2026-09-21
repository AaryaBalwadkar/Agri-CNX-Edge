package com.aarya.agricnxedge.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

// ─────────────────────────────────────────────────────────────────
//  Agri-CNX-Edge · Typography
//
//  Font strategy:
//   • FontFamily.Default (system sans-serif, e.g. Roboto on Android)
//     requires zero asset bundling – safe for a production APK.
//   • Display / Headline roles → SemiBold / Bold for strong hierarchy.
//   • Body & Label roles      → Medium weight for crisp legibility
//     under high ambient light and direct sunlight in the field.
//   • Line heights are slightly generous (+4–6 sp) to aid readability
//     on small-screen devices mounted to agricultural equipment.
// ─────────────────────────────────────────────────────────────────

private val AgriFont = FontFamily.Default

val AgriTypography = Typography(

    // ── Display ─────────────────────────────────────────────────
    displayLarge = TextStyle(
        fontFamily = AgriFont,
        fontWeight = FontWeight.Bold,
        fontSize   = 57.sp,
        lineHeight = 64.sp,
        letterSpacing = (-0.25).sp,
    ),
    displayMedium = TextStyle(
        fontFamily = AgriFont,
        fontWeight = FontWeight.Bold,
        fontSize   = 45.sp,
        lineHeight = 52.sp,
        letterSpacing = 0.sp,
    ),
    displaySmall = TextStyle(
        fontFamily = AgriFont,
        fontWeight = FontWeight.SemiBold,
        fontSize   = 36.sp,
        lineHeight = 44.sp,
        letterSpacing = 0.sp,
    ),

    // ── Headline ─────────────────────────────────────────────────
    headlineLarge = TextStyle(
        fontFamily = AgriFont,
        fontWeight = FontWeight.SemiBold,
        fontSize   = 32.sp,
        lineHeight = 40.sp,
        letterSpacing = 0.sp,
    ),
    headlineMedium = TextStyle(
        fontFamily = AgriFont,
        fontWeight = FontWeight.SemiBold,
        fontSize   = 28.sp,
        lineHeight = 36.sp,
        letterSpacing = 0.sp,
    ),
    headlineSmall = TextStyle(
        fontFamily = AgriFont,
        fontWeight = FontWeight.SemiBold,
        fontSize   = 24.sp,
        lineHeight = 32.sp,
        letterSpacing = 0.sp,
    ),

    // ── Title ─────────────────────────────────────────────────────
    titleLarge = TextStyle(
        fontFamily = AgriFont,
        fontWeight = FontWeight.SemiBold,
        fontSize   = 22.sp,
        lineHeight = 28.sp,
        letterSpacing = 0.sp,
    ),
    titleMedium = TextStyle(
        fontFamily = AgriFont,
        fontWeight = FontWeight.Medium,
        fontSize   = 16.sp,
        lineHeight = 24.sp,
        letterSpacing = 0.15.sp,
    ),
    titleSmall = TextStyle(
        fontFamily = AgriFont,
        fontWeight = FontWeight.Medium,
        fontSize   = 14.sp,
        lineHeight = 20.sp,
        letterSpacing = 0.1.sp,
    ),

    // ── Body ──────────────────────────────────────────────────────
    // Medium weight (W500) ensures strokes are thick enough to remain
    // legible under glare, without crossing into heavy/Bold territory
    // that would feel aggressive at paragraph length.
    bodyLarge = TextStyle(
        fontFamily    = AgriFont,
        fontWeight    = FontWeight.Medium,   // ← field-readability key
        fontSize      = 16.sp,
        lineHeight    = 26.sp,              // +2 sp vs M3 default
        letterSpacing = 0.5.sp,
    ),
    bodyMedium = TextStyle(
        fontFamily    = AgriFont,
        fontWeight    = FontWeight.Medium,   // ← field-readability key
        fontSize      = 14.sp,
        lineHeight    = 22.sp,
        letterSpacing = 0.25.sp,
    ),
    bodySmall = TextStyle(
        fontFamily    = AgriFont,
        fontWeight    = FontWeight.Medium,   // ← field-readability key
        fontSize      = 12.sp,
        lineHeight    = 18.sp,
        letterSpacing = 0.4.sp,
    ),

    // ── Label ─────────────────────────────────────────────────────
    labelLarge = TextStyle(
        fontFamily    = AgriFont,
        fontWeight    = FontWeight.SemiBold,
        fontSize      = 14.sp,
        lineHeight    = 20.sp,
        letterSpacing = 0.1.sp,
    ),
    labelMedium = TextStyle(
        fontFamily    = AgriFont,
        fontWeight    = FontWeight.Medium,
        fontSize      = 12.sp,
        lineHeight    = 16.sp,
        letterSpacing = 0.5.sp,
    ),
    labelSmall = TextStyle(
        fontFamily    = AgriFont,
        fontWeight    = FontWeight.Medium,
        fontSize      = 11.sp,
        lineHeight    = 16.sp,
        letterSpacing = 0.5.sp,
    ),
)