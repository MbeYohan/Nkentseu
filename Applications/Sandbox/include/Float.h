// ArMath/Float.h — Utilitaires arithmetique flottante IEEE 754
#pragma once

#include <cmath>
#include <limits>
#include <cstdint>
#include <cstring>
#include <vector>
#include <bitset>

#include "NKLogger/NkLog.h"
#include "NKContainers/String/NkStringUtils.h"

namespace NkMath {

    // Seuils de comparaison
    constexpr double kEps  = 1e-9;
    constexpr float  kFEps = 1e-6f;

    // --- Validite flottante ---
    inline bool isFiniteValid(double v) { return std::isfinite(v); }
    inline bool isFiniteValid(float  v) { return std::isfinite(v); }

    // --- Proche de zero ---
    inline bool nearlyZero(double v, double tol = kEps)  { return std::fabs(v) < tol; }
    inline bool nearlyZero(float  v, float  tol = kFEps) { return std::fabs(v) < tol; }

    // --- Egalite relative ---
    inline bool approxEq(double a, double b, double tol = kEps) {
        if (a == b) return true;
        double ref = std::max(std::fabs(a), std::fabs(b));
        return std::fabs(a - b) <= tol * std::max(1.0, ref);
    }

    // --- Sommation compensee de Kahan (float) ---
    inline float kahanSum(std::vector<float>& vals) {
        float total = 0.0f;
        float err   = 0.0f;
        for (size_t i = 0; i < vals.size(); i++) {
            float corrected = vals[i] - err;
            float newTotal  = total + corrected;
            err   = (newTotal - total) - corrected;
            total = newTotal;
        }
        return total;
    }

    // --- Sommation compensee de Kahan (double vers float) ---
    inline float kahanSum(std::vector<double>& vals) {
        float total = 0.0f;
        float err   = 0.0f;
        for (size_t i = 0; i < vals.size(); i++) {
            float corrected = (float)vals[i] - err;
            float newTotal  = total + corrected;
            err   = (newTotal - total) - corrected;
            total = newTotal;
        }
        return total;
    }

    // --- Inspection binaire d'un float (IEEE 754 32 bits) ---
    // valeur = (-1)^signe x 1.mantisse x 2^(exposant - 127)
    inline void inspectFloat(float val) {
        uint32_t raw;
        std::memcpy(&raw, &val, sizeof(raw));

        uint32_t sgn = (raw >> 31) & 0x1;
        uint32_t exp = (raw >> 23) & 0xFF;
        uint32_t man = raw & 0x7FFFFF;

        logger.Info(
            "Float {0} en binaire IEEE 754 :\n- Signe    : {1}\n- Exposant : {2}\n- Mantisse : {3}",
            val, sgn,
            std::bitset<8>(exp).to_string().c_str(),
            std::bitset<23>(man).to_string().c_str()
        );
    }

    // --- Inspection binaire d'un double (IEEE 754 64 bits) ---
    // valeur = (-1)^s x 1.mantisse x 2^(exposant - 1023)
    inline void inspectDouble(double val) {
        uint64_t raw;
        std::memcpy(&raw, &val, sizeof(raw));

        uint64_t sgn = (raw >> 63) & 0x1;
        uint64_t exp = (raw >> 52) & 0x7FF;
        uint64_t man = raw & 0xFFFFFFFFFFFFFULL;

        logger.Info(
            "Double {0} en binaire IEEE 754 :\n- Signe    : {1}\n- Exposant : {2}\n- Mantisse : {3}",
            val, sgn,
            std::bitset<11>(exp).to_string().c_str(),
            std::bitset<52>(man).to_string().c_str()
        );
    }

    // --- Variance methode naive (instable) ---
    inline float varianceNaive(const std::vector<float>& data) {
        float s = 0.0f, s2 = 0.0f;
        for (float x : data) { s += x; s2 += x * x; }
        float mu = s / (float)data.size();
        return (s2 / (float)data.size()) - mu * mu;
    }

    // --- Variance methode de Welford (stable numeriquement) ---
    inline float varianceWelford(const std::vector<float>& data) {
        float mu = 0.0f, M2 = 0.0f;
        int   cnt = 0;
        for (float x : data) {
            cnt++;
            float d1 = x - mu;
            mu += d1 / (float)cnt;
            float d2 = x - mu;
            M2 += d1 * d2;
        }
        return M2 / (float)cnt;
    }

    // --- Epsilon machine par dichotomie ---
    inline float epsilonMachine() {
        float e = 1.0f;
        while ((1.0f + e * 0.5f) > 1.0f) e *= 0.5f;
        return e;
    }

} // namespace NkMath
