// ArMath/Color.h — Interpolation bilineaire sub-pixel
#pragma once
#include "NKImage.h"

namespace NkMath {

    struct Color4 { double r, g, b, a; };

    // Accès sub-pixel par interpolation bilineaire
    inline Color4 SampleBilinear(const NkImage& img, double fx, double fy) {
        int x0 = (int)fx, y0 = (int)fy;
        int x1 = std::min(x0+1, img.Width()-1);
        int y1 = std::min(y0+1, img.Height()-1);
        x0 = std::max(0,x0); y0 = std::max(0,y0);

        double tx = fx-(int)fx, ty = fy-(int)fy;

        const uint8_t* p00=img.At(x0,y0), *p10=img.At(x1,y0);
        const uint8_t* p01=img.At(x0,y1), *p11=img.At(x1,y1);
        if (!p00||!p10||!p01||!p11) return {0,0,0,255};

        Color4 c;
        for (int i=0;i<4;i++) {
            double top    = p00[i]*(1-tx) + p10[i]*tx;
            double bottom = p01[i]*(1-tx) + p11[i]*tx;
            (&c.r)[i]     = top*(1-ty) + bottom*ty;
        }
        return c;
    }

} // namespace NkMath
