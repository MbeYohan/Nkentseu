// ArMath/IntegralImage.h — Table des aires (SAT) + seuillage adaptatif
#pragma once
#include "Color.h"

namespace NkMath {

    // Summed Area Table : construction O(WH), requete rectangulaire O(1)
    class IntegralImage {
    public:
        IntegralImage(const std::vector<uint8_t>& gray, int w, int h)
            : m_w(w), m_h(h), m_sat(w*h, 0LL)
        {
            for (int y=0; y<h; y++) for (int x=0; x<w; x++) {
                int idx = y*w+x;
                m_sat[idx] = (long long)gray[idx]
                    + (x>0   ? m_sat[idx-1]   : 0LL)
                    + (y>0   ? m_sat[idx-w]   : 0LL)
                    - (x>0&&y>0 ? m_sat[idx-w-1] : 0LL);
            }
        }

        // Somme du rectangle [x0,y0]->[x1,y1] en O(1)
        long long RectSum(int x0, int y0, int x1, int y1) const {
            x0=std::max(0,x0); y0=std::max(0,y0);
            x1=std::min(m_w-1,x1); y1=std::min(m_h-1,y1);
            long long s = m_sat[y1*m_w+x1];
            if (x0>0) s -= m_sat[y1*m_w+(x0-1)];
            if (y0>0) s -= m_sat[(y0-1)*m_w+x1];
            if (x0>0&&y0>0) s += m_sat[(y0-1)*m_w+(x0-1)];
            return s;
        }

        int Area(int x0,int y0,int x1,int y1) const {
            return (x1-x0+1)*(y1-y0+1);
        }

    private:
        int m_w, m_h;
        std::vector<long long> m_sat;
    };

    // Seuillage adaptatif : chaque pixel compare a la moyenne locale
    inline NkImage AdaptiveThreshold(const NkImage& src, int blkSz=31, int offset=7) {
        auto gray = src.ToGrayscale();
        int w=src.Width(), h=src.Height();
        IntegralImage sat(gray,w,h);
        NkImage out(w,h);
        int half = blkSz/2;
        for (int y=0;y<h;y++) for (int x=0;x<w;x++) {
            long long s = sat.RectSum(x-half,y-half,x+half,y+half);
            int area     = sat.Area(x-half,y-half,x+half,y+half);
            int avg      = (int)(s/area);
            uint8_t pix  = gray[y*w+x];
            uint8_t res  = (pix < avg-offset) ? 0 : 255;
            out.SetPixelGray(x,y,res);
        }
        return out;
    }

} // namespace NkMath
