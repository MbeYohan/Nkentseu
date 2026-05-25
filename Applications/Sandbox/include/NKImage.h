// ArMath/NKImage.h — Image RGBA en memoire, I/O PPM, convolution
#pragma once
#include <cstdint>
#include <vector>
#include <cassert>
#include <cstring>
#include <string>
#include <algorithm>
#include <filesystem>

namespace NkMath {

    class NkImage {
    public:
        NkImage() : m_w(0), m_h(0) {}
        NkImage(int w, int h) : m_w(w), m_h(h), m_buf(w*h*4, 0) {}

        // Acces pixel avec verification de bornes
        uint8_t* At(int x, int y) {
            if (x < 0 || x >= m_w || y < 0 || y >= m_h) return nullptr;
            return m_buf.data() + (y*m_w + x)*4;
        }
        const uint8_t* At(int x, int y) const {
            if (x < 0 || x >= m_w || y < 0 || y >= m_h) return nullptr;
            return m_buf.data() + (y*m_w + x)*4;
        }

        void SetPixelRGBA(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a=255) {
            uint8_t* p = At(x,y);
            if (p) { p[0]=r; p[1]=g; p[2]=b; p[3]=a; }
        }
        void SetPixelGray(int x, int y, uint8_t v) { SetPixelRGBA(x,y,v,v,v,255); }

        int Width()  const { return m_w; }
        int Height() const { return m_h; }

        const uint8_t* Data() const { return m_buf.data(); }
        uint8_t*       Data()       { return m_buf.data(); }
        size_t         DataSize() const { return m_buf.size(); }

        void Fill(uint8_t r, uint8_t g, uint8_t b, uint8_t a=255) {
            for (int y=0; y<m_h; y++)
                for (int x=0; x<m_w; x++)
                    SetPixelRGBA(x,y,r,g,b,a);
        }

        // Sauvegarde PPM P6 (sans dependance externe)
        bool SavePPM(const std::string& fname) const {
            std::string path = "generatedImages/" + fname;
            std::filesystem::create_directories("generatedImages");
            FILE* f = fopen(path.c_str(), "wb");
            if (!f) return false;
            fprintf(f, "P6\n%d %d\n255\n", m_w, m_h);
            for (int y=0; y<m_h; y++) {
                for (int x=0; x<m_w; x++) {
                    const uint8_t* p = At(x,y);
                    if (p) { fwrite(p, 1, 3, f); }
                    else   { uint8_t blk[3]={0,0,0}; fwrite(blk,1,3,f); }
                }
            }
            fclose(f);
            return true;
        }

        bool LoadPPM(const std::string& path) {
            FILE* f = fopen(path.c_str(), "rb");
            if (!f) return false;
            char hdr[3]; int w,h,mx;
            fscanf(f, "%2s %d %d %d ", hdr, &w, &h, &mx);
            if (hdr[0]!='P'||hdr[1]!='6') { fclose(f); return false; }
            *this = NkImage(w,h);
            for (int y=0; y<h; y++) for (int x=0; x<w; x++) {
                uint8_t rgb[3]; fread(rgb,1,3,f);
                SetPixelRGBA(x,y,rgb[0],rgb[1],rgb[2],255);
            }
            fclose(f);
            return true;
        }

        // Niveaux de gris (moyenne RGB)
        std::vector<uint8_t> ToGrayscale() const {
            std::vector<uint8_t> gry(m_w*m_h);
            for (int y=0; y<m_h; y++) for (int x=0; x<m_w; x++) {
                const uint8_t* p = At(x,y);
                gry[y*m_w+x] = p ? (p[0]+p[1]+p[2])/3 : 0;
            }
            return gry;
        }

        // Convolution 2D generique
        NkImage Convolve(const std::vector<double>& ker, int kSz) const {
            int half = kSz/2;
            NkImage out(m_w, m_h);
            for (int y=0; y<m_h; y++) {
                for (int x=0; x<m_w; x++) {
                    double acc[4] = {0,0,0,0};
                    for (int ky=-half; ky<=half; ky++) {
                        for (int kx=-half; kx<=half; kx++) {
                            int sx = std::clamp(x+kx,0,m_w-1);
                            int sy = std::clamp(y+ky,0,m_h-1);
                            double kv = ker[(ky+half)*kSz+(kx+half)];
                            const uint8_t* p = At(sx,sy);
                            if (p) {
                                acc[0]+=p[0]*kv; acc[1]+=p[1]*kv;
                                acc[2]+=p[2]*kv; acc[3]+=p[3]*kv;
                            }
                        }
                    }
                    out.SetPixelRGBA(x,y,
                        (uint8_t)std::clamp(acc[0],0.0,255.0),
                        (uint8_t)std::clamp(acc[1],0.0,255.0),
                        (uint8_t)std::clamp(acc[2],0.0,255.0),
                        (uint8_t)std::clamp(acc[3],0.0,255.0));
                }
            }
            return out;
        }

        // Bresenham
        void DrawLine(int x0,int y0,int x1,int y1,
                      uint8_t r=0,uint8_t g=0,uint8_t b=0,uint8_t a=255) {
            int dx=std::abs(x1-x0), dy=-std::abs(y1-y0);
            int sx=(x0<x1)?1:-1, sy=(y0<y1)?1:-1, err=dx+dy;
            while (true) {
                SetPixelRGBA(x0,y0,r,g,b,a);
                if (x0==x1&&y0==y1) break;
                int e2=2*err;
                if (e2>=dy){err+=dy;x0+=sx;}
                if (e2<=dx){err+=dx;y0+=sy;}
            }
        }

        // Combinaison de gradients (magnitude Sobel)
        static NkImage CombineGradient(const NkImage& gx, const NkImage& gy) {
            assert(gx.Width()==gy.Width()&&gx.Height()==gy.Height());
            int w=gx.Width(), h=gx.Height();
            NkImage out(w,h);
            for (int y=0;y<h;y++) for (int x=0;x<w;x++) {
                const uint8_t* px=gx.At(x,y);
                const uint8_t* py=gy.At(x,y);
                if (!px||!py){out.SetPixelGray(x,y,0);continue;}
                double mag=std::sqrt((double)px[0]*px[0]+(double)py[0]*py[0]);
                out.SetPixelGray(x,y,(uint8_t)std::clamp(mag,0.0,255.0));
            }
            return out;
        }

    private:
        int m_w, m_h;
        std::vector<uint8_t> m_buf;   // RGBA contigu
    };

} // namespace NkMath
