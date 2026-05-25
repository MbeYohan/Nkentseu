#include <Unitest/Unitest.h>
#include <Unitest/TestMacro.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <random>

#include "NKLogger/NkLog.h"
#include "NKMath/NKMath.h"
#include "SVD.h"
#include "Quat.h"
#include "IntegralImage.h"

using namespace NkMath;

// --- Donnees globales partagees entre les TPs ---
static const int IMG_W = 512, IMG_H = 512;
static NkImage canvas(IMG_W, IMG_H);

static std::mt19937 rgen(123);
static std::uniform_real_distribution<double> rdist(-10.0, 10.0);

// Sommets du cube unitaire centre en [-0.5, 0.5]^3
static std::vector<Vec4d> cubeVerts = {
    {-0.5,-0.5,-0.5,1}, { 0.5,-0.5,-0.5,1},
    { 0.5, 0.5,-0.5,1}, {-0.5, 0.5,-0.5,1},
    {-0.5,-0.5, 0.5,1}, { 0.5,-0.5, 0.5,1},
    { 0.5, 0.5, 0.5,1}, {-0.5, 0.5, 0.5,1}
};

// 12 aretes du cube (paires d'indices de sommets)
static std::vector<Vec2d> cubeEdges = {
    {0,1},{1,2},{2,3},{3,0},
    {4,5},{5,6},{6,7},{7,4},
    {0,4},{1,5},{2,6},{3,7}
};

// Matrices View et Projection pour le rasteriseur
static Vec3d camPos{0,1,3}, lookAt{0,0,0}, upDir{0,1,0};
static Mat4d viewMat  = LookAt(camPos, lookAt, upDir);
static Mat4d projMat  = Perspective(60.0, (double)IMG_W/IMG_H, 0.1, 100.0);


// ===========================================================================
// SEMAINE 1 — MODULE 01 : IEEE 754 & Arithmetique Flottante
// ===========================================================================

// TP1 : inspectFloat sur les 6 valeurs imposees par le cours
TEST_CASE(Semaine1_TP1, FonctionInspectFloat) {
    inspectFloat(0.1f);                               // 0.1 inexact en binaire
    inspectFloat(1.0f);                               // signe=0, exp=127, mant=0
    inspectFloat(1.0f / 0.0f);                        // +Inf
    inspectFloat(std::sqrt(-1.0f));                   // NaN
    inspectFloat(-0.0f);                              // -0 : bit31=1
    inspectFloat(0.0f);                               // +0 : tous bits nuls
    inspectFloat(std::numeric_limits<float>::min());  // subnormal
}

// TP2 : demonstration empirique des problemes de precision
TEST_CASE(Semaine1_TP2, ProblemsDePrecision) {
    float accSum, kahSum;

    // 1 000 000 copies de 0.1f
    std::vector<float> bigVec(1'000'000, 0.1f);

    accSum = std::accumulate(bigVec.begin(), bigVec.end(), 0.0f);
    kahSum = kahanSum(bigVec);
    logger.Info("\nstd::accumulate : {0}\nkahanSum        : {1}\nValeur theorique : 100000.0",
                accSum, kahSum);

    // Variance naive vs Welford sur valeurs proches de 1e8
    std::vector<float> tricky = {1e8f, 1e8f, 1.0f, 2.0f};
    logger.Info("\nvarianceNaive   : {0}\nvarianceWelford : {1}",
                varianceNaive(tricky), varianceWelford(tricky));

    // Epsilon machine
    logger.Info("\nepsilonMachine() : {0}\nnumeric_limits   : {1}",
                epsilonMachine(), std::numeric_limits<float>::epsilon());
}

// TP3 : 33 tests unitaires sur Float.h
TEST_CASE(Semaine1_TP3, TestsUnitairesFloath) {
    // --- isFiniteValid (5 tests) ---
    ASSERT_TRUE(!isFiniteValid(std::numeric_limits<float>::quiet_NaN())); // 1
    ASSERT_TRUE(!isFiniteValid( std::numeric_limits<float>::infinity())); // 2
    ASSERT_TRUE(!isFiniteValid(-std::numeric_limits<float>::infinity())); // 3
    ASSERT_TRUE( isFiniteValid(0.0f));                                    // 4
    ASSERT_TRUE( isFiniteValid(1.0f));                                    // 5

    // --- nearlyZero (8 tests) ---
    ASSERT_TRUE( nearlyZero( 0.0f,  1e-6f));   // 6
    ASSERT_TRUE(!nearlyZero( 1e-5f, 1e-6f));   // 7
    ASSERT_TRUE( nearlyZero( 1e-7f, 1e-6f));   // 8
    ASSERT_TRUE( nearlyZero(-1e-7f, 1e-6f));   // 9
    ASSERT_TRUE(!nearlyZero(-1e-6f, 1e-7f));   // 10
    ASSERT_TRUE( nearlyZero( 1e-3f, 1e-2f));   // 11
    ASSERT_TRUE(!nearlyZero( 1e-2f, 1e-3f));   // 12
    ASSERT_TRUE( nearlyZero( 5e-8f, 1e-7f));   // 13

    // --- approxEq (10 tests) ---
    ASSERT_TRUE( approxEq(1.0f,  1.0f,      1e-6f));  // 14
    ASSERT_TRUE( approxEq(1.0f,  1.0000001f,1e-5f));  // 15
    ASSERT_TRUE(!approxEq(1.0f,  1.1f,      1e-3f));  // 16
    ASSERT_TRUE( approxEq(0.0f,  1e-7f,     1e-6f));  // 17
    ASSERT_TRUE(!approxEq(0.0f,  1e-4f,     1e-6f));  // 18
    ASSERT_TRUE( approxEq(-1.0f,-1.000001f, 1e-5f));  // 19
    ASSERT_TRUE(!approxEq(-1.0f,-1.1f,      1e-2f));  // 20
    ASSERT_TRUE( approxEq(1000.0f,1000.0001f,1e-3f)); // 21
    ASSERT_TRUE( approxEq(1000.0f,1001.0f,  1e-3f));  // 22
    ASSERT_TRUE( approxEq(1e-7f, 2e-7f,     1e-6f));  // 23

    // --- kahanSum (10 tests) ---
    float s1, s2;
    std::vector<float> v;

    v={std::vector<float>(1000,0.1f)};
    s1=std::accumulate(v.begin(),v.end(),0.0f); s2=kahanSum(v);
    ASSERT_TRUE(std::fabs(s2-100.0f)<std::fabs(s1-100.0f));  // 24

    v={std::vector<float>(10000,0.1f)};
    s1=std::accumulate(v.begin(),v.end(),0.0f); s2=kahanSum(v);
    ASSERT_TRUE(std::fabs(s2-1000.0f)<std::fabs(s1-1000.0f)); // 25

    v={1e8f,1.0f,-1e8f};
    s1=std::accumulate(v.begin(),v.end(),0.0f); s2=kahanSum(v);
    ASSERT_TRUE(std::fabs(s2-1.0f)<=std::fabs(s1-1.0f));      // 26

    v={1.0f,1e8f,-1e8f};
    s1=std::accumulate(v.begin(),v.end(),0.0f); s2=kahanSum(v);
    ASSERT_TRUE(std::fabs(s2-1.0f)<=std::fabs(s1-1.0f));      // 27

    v=std::vector<float>(100000,0.01f); s2=kahanSum(v);
    ASSERT_TRUE(approxEq(s2,1000.0f,1e-2f));                   // 28

    v=std::vector<float>(100000,1e-5f); s2=kahanSum(v);
    ASSERT_TRUE(approxEq(s2,1.0f,1e-3f));                      // 29

    v={0.1f,0.2f,0.3f}; s2=kahanSum(v);
    ASSERT_TRUE(approxEq(s2,0.6f,1e-6f));                      // 30

    v=std::vector<float>(50000,0.2f); s2=kahanSum(v);
    ASSERT_TRUE(approxEq(s2,10000.0f,1e-2f));                  // 31

    v={1e7f,1.0f,1.0f,-1e7f}; s2=kahanSum(v);
    ASSERT_TRUE(approxEq(s2,2.0f,1e-3f));                      // 32

    v=std::vector<float>(1000000,0.1f); s2=kahanSum(v);
    ASSERT_TRUE(approxEq(s2,100000.0f,1e-1f));                 // 33
}


// ===========================================================================
// SEMAINE 2 — MODULE 02 : Vecteurs Vec2d, Vec3d, Vec4d
// ===========================================================================

// TP4 : Vec2d complet avec 20 assertions
TEST_CASE(Semaine2_TP1, Vec2dComplet) {
    // Dot product (6 tests)
    ASSERT_TRUE(Dot(Vec2d{1,0},Vec2d{0,1}) == 0.0);   // 1
    ASSERT_TRUE(Dot(Vec2d{1,0},Vec2d{1,0}) == 1.0);   // 2
    ASSERT_TRUE(Dot(Vec2d{3,4},Vec2d{3,4}) == 25.0);  // 3
    ASSERT_TRUE(Dot(Vec2d{-1,0},Vec2d{1,0}) == -1.0); // 4
    ASSERT_TRUE(Dot(Vec2d{2,3},Vec2d{4,5}) == 23.0);  // 5
    ASSERT_TRUE(Dot(Vec2d{0,0},Vec2d{5,7}) == 0.0);   // 6

    // Cross2D (4 tests)
    ASSERT_TRUE(Cross2D(Vec2d{1,0},Vec2d{0,1}) ==  1.0); // 7
    ASSERT_TRUE(Cross2D(Vec2d{0,1},Vec2d{1,0}) == -1.0); // 8
    ASSERT_TRUE(Cross2D(Vec2d{1,1},Vec2d{1,1}) ==  0.0); // 9
    ASSERT_TRUE(Cross2D(Vec2d{2,0},Vec2d{0,2}) ==  4.0); // 10

    // Normalisation (4 tests)
    Vec2d n = Vec2d{3,4}.Normalized();
    ASSERT_TRUE(std::fabs(n.Norm()-1.0) < kEps);  // 11
    ASSERT_TRUE(std::fabs(n.x-0.6)      < kEps);  // 12
    ASSERT_TRUE(std::fabs(n.y-0.8)      < kEps);  // 13
    Vec2d u = Vec2d{1,0}.Normalized();
    ASSERT_TRUE(std::fabs(u.x-1.0)      < kEps);  // 14

    // Operateur [] (5 tests)
    Vec2d w{10,20};
    ASSERT_TRUE(w[0] == 10.0); // 15
    ASSERT_TRUE(w[1] == 20.0); // 16
    w[0] = 30; ASSERT_TRUE(w.x == 30.0); // 17
    w[1] = 40; ASSERT_TRUE(w.y == 40.0); // 18
    Vec2d q{5,6}; ASSERT_TRUE(q[0] == 5.0); // 19

    // Layout memoire (static_assert = 1 test)
    static_assert(sizeof(Vec2d) == 16, "Vec2d doit faire 16 octets"); // 20
}

// TP5 : Vec3d + cross product + Gram-Schmidt
TEST_CASE(Semaine2_TP2, Vec3dEtGramSchmidt) {
    Vec3d ex{1,0,0}, ey{0,1,0}, ez{0,0,1};

    // Cross product : regle main droite (6 tests)
    ASSERT_TRUE(ApproxVec(Cross(ex,ey), ez));           // 1
    ASSERT_TRUE(ApproxVec(Cross(ey,ex), {0,0,-1}));     // 2
    ASSERT_TRUE(ApproxVec(Cross(ey,ez), ex));           // 3
    ASSERT_TRUE(ApproxVec(Cross(ez,ex), ey));           // 4
    ASSERT_TRUE(approxEq(Dot(Cross(ex,ey),ex), 0.0));  // 5
    ASSERT_TRUE(approxEq(Dot(Cross(ex,ey),ey), 0.0));  // 6

    // Gram-Schmidt sur 10 triplets aleatoires (6 assertions par triplet)
    for (int t=0; t<10; t++) {
        Vec3d a{rdist(rgen),rdist(rgen),rdist(rgen)};
        Vec3d b{rdist(rgen),rdist(rgen),rdist(rgen)};
        Vec3d c{rdist(rgen),rdist(rgen),rdist(rgen)};

        Vec3d ui = a.Normalized();
        Vec3d vi = (b - Project(b,ui)).Normalized();
        Vec3d wi = (c - Project(c,ui) - Project(c,vi)).Normalized();

        ASSERT_TRUE(approxEq(ui.Norm(), 1.0)); // 7
        ASSERT_TRUE(approxEq(vi.Norm(), 1.0)); // 8
        ASSERT_TRUE(approxEq(wi.Norm(), 1.0)); // 9
        ASSERT_TRUE(approxEq(Dot(ui,vi), 0.0)); // 10
        ASSERT_TRUE(approxEq(Dot(ui,wi), 0.0)); // 11
        ASSERT_TRUE(approxEq(Dot(vi,wi), 0.0)); // 12
    }

    // Project + Reject
    Vec3d a{3,4,0}, b{1,0,0};
    ASSERT_TRUE(ApproxVec(Project(a,b)+Reject(a,b), a)); // 13
}

// TP6 : Vec4d + projection perspective + image PPM
TEST_CASE(Semaine2_TP3, Vec4dEtProjectionPerspective) {
    std::vector<Vec2d> pts2D;
    double zCam = 2.0;

    // Deplacer le cube devant la camera et projeter
    auto verts = cubeVerts;
    for (auto& p : verts) {
        p.z += zCam;
        pts2D.push_back(ProjectPoint(p));
    }

    // Dessiner les coins (carres rouges 5x5)
    for (const auto& p : pts2D) {
        int px=(int)p.x, py=(int)p.y;
        for (int dy=-2;dy<=2;dy++) for (int dx=-2;dx<=2;dx++)
            canvas.SetPixelRGBA(px+dx,py+dy,255,0,0);
    }

    // Dessiner les 12 aretes (noir)
    for (auto& e : cubeEdges) {
        int i=(int)e.x, j=(int)e.y;
        canvas.DrawLine((int)pts2D[i].x,(int)pts2D[i].y,
                        (int)pts2D[j].x,(int)pts2D[j].y);
    }
    canvas.SavePPM("cube_projection.ppm");
}


// ===========================================================================
// SEMAINE 3 — MODULE 03 : Matrices Mat3d, Mat4d
// ===========================================================================

// TP7 : Mat4d + Inverse + Rodrigues
TEST_CASE(Semaine3_TP1, Mat4dEtInverse) {
    Mat4d M, Inv, Prod;
    std::uniform_real_distribution<double> ud(-5.0,5.0);

    for (int t=0; t<10; t++) {
        for (int i=0;i<4;i++) for (int j=0;j<4;j++) M(i,j)=ud(rgen);

        // M * I == M
        Prod = M * Mat4d::Identity();
        ASSERT_TRUE(ApproxMat(Prod, M)); // 1-10

        // M * M^-1 == I
        if (Inverse(M, Inv))
            ASSERT_TRUE(ApproxMat(M*Inv, Mat4d::Identity(), 1e-10)); // 11-20
    }

    // Matrice singuliere -> Inverse retourne false
    Mat4d Sing = Mat4d::Identity();
    for (int j=0;j<4;j++) Sing(1,j)=Sing(0,j);  // ligne dupliquee
    ASSERT_TRUE(!Inverse(Sing, Inv)); // 21

    // Rodrigues : RotY(90) * (1,0,0,1) = (0,0,-1,1)
    Mat4d Ry = Mat4d::RotateAxis({0,1,0}, NKENTSEU_PI_DOUBLE/2.0);
    Vec4d res = Ry * Vec4d{1,0,0,1};
    ASSERT_TRUE(approxEq(res.x,  0.0)); // 22
    ASSERT_TRUE(approxEq(res.y,  0.0)); // 23
    ASSERT_TRUE(approxEq(res.z, -1.0)); // 24
}

// TP8 : Rasteriseur logiciel avec LookAt — 10 frames PPM
TEST_CASE(Semaine3_TP2, RasteriseurAvecLookAt) {
    for (int frame=0; frame<10; frame++) {
        canvas.Fill(255,255,255);  // fond blanc

        double ang = frame * 0.3;
        Mat4d R = Mat4d::RotateAxis(upDir, ang);
        std::vector<Vec3d> screen;

        for (auto v : cubeVerts) {
            Vec4d clip = projMat * (viewMat * (R * v));
            screen.push_back(ProjectToScreen(clip, IMG_W, IMG_H));
        }

        for (auto& e : cubeEdges) {
            int i=(int)e.x, j=(int)e.y;
            canvas.DrawLine((int)screen[i].x,(int)screen[i].y,
                            (int)screen[j].x,(int)screen[j].y, 255,0,0);
        }
        canvas.SavePPM("rast_frame_"+std::to_string(frame)+".ppm");
    }
}

// TP9 : TRS 3D + Decomposition
TEST_CASE(Semaine3_TP3, TRS3DEtDecomposition) {
    std::uniform_real_distribution<double> td(-5.0,5.0);
    std::uniform_real_distribution<double> sd( 6.0,9.0);  // scale >0

    for (int t=0; t<20; t++) {
        Vec3d inT{td(rgen),td(rgen),td(rgen)};
        Vec3d inR{td(rgen),td(rgen),td(rgen)};
        Vec3d inS{sd(rgen),sd(rgen),sd(rgen)};

        Mat4d M = TRS(inT, inR, inS);

        Vec3d outT, outR, outS;
        DecomposeTRS(M, outT, outR, outS);

        ASSERT_TRUE(ApproxVec(inT, outT));        // translation exacte
        ASSERT_TRUE(ApproxVec(inS, outS));        // echelle exacte
        ASSERT_TRUE(ApproxVec(inR, outR, 5.0));  // rotation (tolerance large)
    }
}


// ===========================================================================
// SEMAINE 4 — MODULE 04 : Quaternions
// ===========================================================================

// TP10 : Quaternions complets
TEST_CASE(Semaine4_TP1, QuaternionsComplets) {
    // 1. Rotate + FromAxisAngle
    Quat qy = FromAxisAngle({0,1,0}, NKENTSEU_PI_DOUBLE/2.0);
    Vec3d r  = Rotate(qy, {1,0,0});
    ASSERT_TRUE(std::fabs(r.x - 0.0) < kEps);  // x = 0
    ASSERT_TRUE(std::fabs(r.y - 0.0) < kEps);  // y = 0
    ASSERT_TRUE(std::fabs(r.z + 1.0) < kEps);  // z = -1

    // 2. Aller-retour Quat -> Mat3 -> Quat (50 quaternions aleatoires)
    std::uniform_real_distribution<double> ud(-1.0,1.0);
    for (int t=0; t<50; t++) {
        Quat q1{ud(rgen),ud(rgen),ud(rgen),ud(rgen)};
        q1 = q1.Normalized();
        Mat3d Rm = ToMat3(q1);
        Quat q2  = FromMat3(Rm).Normalized();
        ASSERT_TRUE(ApproxQuat(q1, q2, 1e-4));
    }

    // 3. q * q.Inverse() = identite (50 quaternions)
    for (int t=0; t<50; t++) {
        Quat q{ud(rgen),ud(rgen),ud(rgen),ud(rgen)};
        q = q.Normalized();
        Quat qi = q.Inverse();
        Quat id = q * qi;
        ASSERT_TRUE(ApproxQuat(id, Quat::Identity(), 1e-4));
    }
}

// TP11 : Animation SLERP vs LERP — 60 frames chacun
TEST_CASE(Semaine4_TP2, AnimationSLERP) {
    Quat qa = FromAxisAngle({0,1,0}, 0.0);
    Quat qb = FromAxisAngle({0,1,0}, NKENTSEU_PI_DOUBLE);

    // SLERP : vitesse angulaire constante
    for (int frame=0; frame<60; frame++) {
        double t  = frame / 59.0;
        Quat q    = Slerp(qa, qb, t);
        Mat4d R   = FromRT(ToMat3(q), {0,0,0});

        canvas.Fill(255,255,255);
        std::vector<Vec3d> sc;
        for (auto v : cubeVerts) sc.push_back(ProjectToScreen(projMat*(viewMat*(R*v)), IMG_W, IMG_H));
        for (auto& e : cubeEdges)
            canvas.DrawLine((int)sc[(int)e.x].x,(int)sc[(int)e.x].y,
                            (int)sc[(int)e.y].x,(int)sc[(int)e.y].y, 255,0,0);
        canvas.SavePPM("slerp_frame_"+std::to_string(frame)+".ppm");
    }

    // LERP : pour comparaison (vitesse non uniforme)
    for (int frame=0; frame<60; frame++) {
        double t  = frame / 59.0;
        Quat q    = Lerp(qa, qb, t);
        Mat4d R   = FromRT(ToMat3(q), {0,0,0});

        canvas.Fill(255,255,255);
        std::vector<Vec3d> sc;
        for (auto v : cubeVerts) sc.push_back(ProjectToScreen(projMat*(viewMat*(R*v)), IMG_W, IMG_H));
        for (auto& e : cubeEdges)
            canvas.DrawLine((int)sc[(int)e.x].x,(int)sc[(int)e.x].y,
                            (int)sc[(int)e.y].x,(int)sc[(int)e.y].y, 0,0,255);
        canvas.SavePPM("lerp_frame_"+std::to_string(frame)+".ppm");
    }
}


// ===========================================================================
// SEMAINE 5 — MODULE 05 : SVD
// ===========================================================================

// TP12 : Tests SVD 3x3
TEST_CASE(Semaine5_TP1, TestsSVD) {
    Mat3d A{}, S{}, Rec{}, Iu{}, Iv{};
    std::uniform_real_distribution<double> ud(-1.0,1.0);

    // 20 matrices aleatoires : reconstruction, orthogonalite U/V, sigma tries
    for (int i=0; i<20; i++) {
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) A(r,c)=ud(rgen);

        SVD3x3 s = svd3x3(A);

        S(0,0)=s.sigma.x; S(1,1)=s.sigma.y; S(2,2)=s.sigma.z;
        Rec = s.U * S * s.V.Transposed();
        ASSERT_TRUE((Rec-A).norm() < 1e-6);          // reconstruction

        Iu = s.U * s.U.Transposed();
        Iv = s.V * s.V.Transposed();
        ASSERT_TRUE((Iu-Mat3d::Identity()).norm() < 1e-6); // U orthogonale
        ASSERT_TRUE((Iv-Mat3d::Identity()).norm() < 1e-6); // V orthogonale

        ASSERT_TRUE(s.sigma.x >= s.sigma.y);  // sigma tris decroissant
        ASSERT_TRUE(s.sigma.y >= s.sigma.z);
    }

    // Matrice rang 1 : les deux dernieres valeurs singulieres sont ~0
    A.setRow(0,{1,2,3}); A.setRow(1,{2,4,6}); A.setRow(2,{3,6,9});
    SVD3x3 sr = svd3x3(A);
    ASSERT_TRUE(std::fabs(sr.sigma.z) < 1e-6);

    // Pseudo-inverse : Ax = b aux moindres carres
    A.setRow(0,{1,2,0}); A.setRow(1,{3,4,0}); A.setRow(2,{5,6,0});
    Vec3d b{7,8,9};
    SVD3x3 sp = svd3x3(A);
    Vec3d x   = sp.pseudoInverse() * b;
    Vec3d res = A*x - b;
    ASSERT_TRUE(res.Norm() < 1e-5);
}

// TP13 : Homographie (non couverte dans le support — placeholder)
TEST_CASE(Semaine5_TP2, ResolutionHomographie) {
    (void)0;  // non traitee dans le cours fourni (camarade l'a aussi omise)
}


// ===========================================================================
// SEMAINE 6 — MODULE 06 : NkImage
// ===========================================================================

// TP14 : NkImage — degrade + formes geometriques
TEST_CASE(Semaine6_TP1, NkImageDeBase) {
    NkImage img(IMG_W, IMG_H);

    // Degrade RGB
    for (int y=0;y<IMG_H;y++) for (int x=0;x<IMG_W;x++) {
        uint8_t r=(uint8_t)(255.0*x/IMG_W);
        uint8_t g=(uint8_t)(255.0*y/IMG_H);
        img.SetPixelRGBA(x,y,r,g,128);
    }

    // Rectangle rouge
    for (int y=100;y<200;y++) for (int x=100;x<300;x++)
        img.SetPixelRGBA(x,y,255,0,0);

    // Diagonale verte
    for (int i=0;i<IMG_W;i++) img.SetPixelRGBA(i,i,0,255,0);

    // Cercle bleu
    int cx=256,cy=256,cr=80;
    for (int y=0;y<IMG_H;y++) for (int x=0;x<IMG_W;x++) {
        int dx=x-cx,dy=y-cy;
        if (dx*dx+dy*dy < cr*cr) img.SetPixelRGBA(x,y,0,0,255);
    }

    img.SavePPM("degrade_formes.ppm");
}

// TP15 : Convolution Gauss 5x5 + Sobel + mesure temps
TEST_CASE(Semaine6_TP2, ConvolutionEtSobel) {
    std::vector<double> gauss5 = {
         1, 4, 6, 4, 1,
         4,16,24,16, 4,
         6,24,36,24, 6,
         4,16,24,16, 4,
         1, 4, 6, 4, 1
    };
    for (auto& v : gauss5) v /= 256.0;

    std::vector<double> sobelX = {-1,0,1,-2,0,2,-1,0,1};
    std::vector<double> sobelY = {-1,-2,-1,0,0,0,1,2,1};

    NkImage src;
    src.LoadPPM("input_arUCO.ppm");

    auto t0 = std::chrono::high_resolution_clock::now();

    NkImage blurred = src.Convolve(gauss5, 5);
    NkImage gx      = blurred.Convolve(sobelX, 3);
    NkImage gy      = blurred.Convolve(sobelY, 3);
    NkImage mag     = NkImage::CombineGradient(gx, gy);

    auto t1 = std::chrono::high_resolution_clock::now();
    mag.SavePPM("sobel_gradient.ppm");

    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    logger.Info("Gauss5x5 + Sobel3x3 + magnitude : {0} ms", ms);
}

// TP16 : Image integrale + seuillage adaptatif
TEST_CASE(Semaine6_TP3, ImageIntegraleEtSeuillage) {
    NkImage src;
    src.LoadPPM("input_arUCO.ppm");

    // Seuillage adaptatif (blockSize=31, offset=7)
    NkImage binaire = AdaptiveThreshold(src, 31, 7);
    binaire.SavePPM("seuillage_adaptatif.ppm");

    // Mesure du temps de construction de la SAT
    auto t0 = std::chrono::high_resolution_clock::now();
    IntegralImage sat(src.ToGrayscale(), src.Width(), src.Height());
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    logger.Info("Construction image integrale ({0}x{1}) : {2} ms",
                src.Width(), src.Height(), ms);
}
