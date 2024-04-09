/* This file was automatically generated by CasADi 3.6.5.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) default_expl_ode_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_c0 CASADI_PREFIX(c0)
#define casadi_c1 CASADI_PREFIX(c1)
#define casadi_c2 CASADI_PREFIX(c2)
#define casadi_c3 CASADI_PREFIX(c3)
#define casadi_c4 CASADI_PREFIX(c4)
#define casadi_c5 CASADI_PREFIX(c5)
#define casadi_c6 CASADI_PREFIX(c6)
#define casadi_c7 CASADI_PREFIX(c7)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[3] = {0, 0, 0};

static const casadi_real casadi_c0[3] = {0., 0., -9.8100000000000005e+00};
static const casadi_real casadi_c1[27] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
static const casadi_real casadi_c2[81] = {3.3333333333358439e-01, 3.3333333333339965e-01, 3.3333333333346360e-01, 2.2026824808563106e-13, 3.4106051316484809e-13, -9.2370555648813024e-13, -4.6629367034256575e-15, -6.2172489379008766e-15, 5.2735593669694936e-16, 3.3333333333358439e-01, 3.3333333333339965e-01, 3.3333333333346360e-01, 2.2026824808563106e-13, 3.4106051316484809e-13, -9.2370555648813024e-13, -4.6629367034256575e-15, -6.2172489379008766e-15, 5.2735593669694936e-16, 3.3333333333358439e-01, 3.3333333333339965e-01, 3.3333333333346360e-01, 2.2026824808563106e-13, 3.4106051316484809e-13, -9.2370555648813024e-13, -4.6629367034256575e-15, -6.2172489379008766e-15, 5.2735593669694936e-16, 1.4180058906487830e+01, 1.4180058905905753e+01, 1.4180058905960323e+01, 3.9823367872767485e+03, 1.0534569403697606e+02, 1.9652558036788832e+01, -6.5088620210474346e-02, -6.6615263202152164e-02, -2.3230201233417347e-03, -3.3650019717715622e+00, -3.3650019724263984e+00, -3.3650019726010214e+00, -1.1513888880661398e+02, 3.9653816504046699e+03, -5.2698615486151539e+00, 1.4274839425638675e-01, 1.2093367374291120e-01, 2.0696348357660099e-02, 3.4833601858523434e+02, 3.4833601858276143e+02, 3.4833601858304883e+02, 5.2470251483720585e+02, 1.3182856188407104e+03, -6.5697185904112484e+02, -2.5563055777055752e+01, -2.5566312927072801e+01, 1.8361861279951093e+00, 2.2229471222979110e+07, 2.2229471223043844e+07, 2.2229471223030325e+07, -4.6942849057738725e+06, -3.0362661939551674e+07, 6.7496807622176424e+07, 4.6463524948711777e+06, 6.5648649208532425e+05, 1.1692066332465583e+05, -3.8609915734593302e+07, -3.8609915734648228e+07, -3.8609915734631598e+07, 6.3501557951780409e+05, 3.3279321041988749e+07, -7.2909607992567018e+07, -6.6276751650819159e+05, 3.3268819397835932e+06, -1.2336957814182715e+05, 2.4053939642531894e+07, 2.4053939642680269e+07, 2.4053939642656665e+07, -3.0828670485956624e+07, -5.8816350743494056e+07, 1.4062419769420710e+08, 1.2340406488460377e+06, 1.2327859246656036e+06, 3.8616094713562061e+06};
static const casadi_real casadi_c3[117] = {-7.8825834748386114e-15, -7.8825834748386114e-15, -7.8825834748386114e-15, -2.4568493542318492e-01, 5.5599527953756933e-01, -2.1087351017176786e+01, 5.0817489408320223e+05, -4.6147590720411693e+05, 1.0900523964849215e+06, -7.5495165674510645e-15, -7.5495165674510645e-15, -7.5495165674510645e-15, -3.0262533410888182e-01, 1.8768081901328060e-01, -3.7461170412292802e+01, 8.2582043926034006e+05, -8.8651781829602341e+05, 1.7438338754338273e+06, -2.5413005033669833e-13, -2.5413005033669833e-13, -2.5413005033669833e-13, -9.9527832123754933e-01, -4.4323992459919737e-01, -9.4367435727735347e+02, -3.9902581524551963e+04, 8.3317144830969046e+04, 7.0613848664758634e+06, -3.6948222259525210e-13, -3.6948222259525210e-13, -3.6948222259525210e-13, -3.7619002661864215e+01, 9.6119062670768471e+00, -7.0167924728157595e+02, -5.3316542170078300e+07, 1.0133986343332620e+08, -5.5280775863474004e+07, 8.3488771451811772e-14, 8.3488771451811772e-14, 8.3488771451811772e-14, 6.1864905997105780e+01, -2.6081109963310155e+01, 3.2884976807237717e+02, -8.0746794827321824e+06, 8.7304373476444762e+06, -1.7413027177903220e+07, -1.0658141036401503e-14, -1.0658141036401503e-14, -1.0658141036401503e-14, 2.5994343983801969e+01, 5.4441337005818241e+01, -1.6628974345961160e+02, 7.2662323563515255e+06, -5.9466960742955711e+06, 3.6025307045894582e+06, 3.6415315207705135e-13, 3.6415315207705135e-13, 3.6415315207705135e-13, -4.6064623505480995e-01, 5.9097536578519794e-01, 7.9207925956945928e+02, -6.6421563994395398e+06, -9.2360560262067104e+05, -2.3808460175612714e+07, -1.9539925233402755e-13, -1.9539925233402755e-13, -1.9539925233402755e-13, -1.9830998907785779e+03, 1.1438010475428382e+02, -4.8108628277406115e+02, 3.7414313880249499e+06, 3.3516127435321175e+05, 2.9074204365115434e+07, -3.1263880373444408e-13, -3.1263880373444408e-13, -3.1263880373444408e-13, -1.0509191622682920e+02, -1.9653354101338409e+03, -1.3241335096522880e+03, 3.0201575131048467e+07, -3.3051901767327476e+07, 5.9462268306041345e+07, 1.1510792319313623e-12, 1.1510792319313623e-12, 1.1510792319313623e-12, -1.9322406477673212e+01, 4.9922785602684598e+00, 2.3219799473228632e+03, -6.5897640045267016e+07, 7.1182460615864530e+07, -1.3503213225334820e+08, -5.2666204730655863e-15, -5.2666204730655863e-15, -5.2666204730655863e-15, -5.5606126920500571e+00, -8.9615758234674701e+00, -7.4858161309997602e+00, -3.4497794764485996e+05, 3.9060453619819705e+05, 5.6181794123191549e+05, 3.8857805861880479e-15, 3.8857805861880479e-15, 3.8857805861880479e-15, 9.2101610601733341e+00, -5.4324028425752999e+00, 6.2153430039170559e+00, 2.0953326675015542e+05, -2.3496500641741476e+05, -3.1860265078198060e+05, 2.6290081223123707e-13, 2.6290081223123707e-13, 2.6290081223123707e-13, 6.9601545609675668e+00, -1.5692874794618547e+01, 5.1197290869316566e+02, -2.2317325070484951e+07, 3.1032920145044424e+07, -4.2213136347539462e+07};
static const casadi_real casadi_c4[36] = {4.2632564145606011e-13, 4.2632564145606011e-13, 4.2632564145606011e-13, -4.8600916040995799e+00, -3.3022634551918600e+01, 2.1095030361287972e+03, -6.4697428387618259e+07, 4.9956373016881406e+07, -6.9073009893294498e+07, -3.7125857943465235e-13, -3.7125857943465235e-13, -3.7125857943465235e-13, 3.6939180386743828e+01, 1.6452452913530578e+01, -1.0857486535253465e+03, 7.6355750223764163e+06, 1.2093085406239884e+07, 4.8393893553795129e+07, 5.2224891078367364e-13, 5.2224891078367364e-13, 5.2224891078367364e-13, 1.3022169182875587e+01, 3.6822043090516672e+01, 9.1730574650084736e+02, -1.0122700570211548e+07, -9.5185719523508493e+06, -5.2409520952779859e+07, 1.1368683772161603e-13, 1.1368683772161603e-13, 1.1368683772161603e-13, -6.1399186084334360e+01, -1.6666777591301070e+01, 4.4125001689279998e+02, 1.4000444092162680e+07, 4.9602556471085530e+06, -2.1062481887530163e+07};
static const casadi_real casadi_c5[4] = {0., 2.3499999999999999e-01, 0., -2.3499999999999999e-01};
static const casadi_real casadi_c6[4] = {2.3499999999999999e-01, 0., -2.3499999999999999e-01, 0.};
static const casadi_real casadi_c7[4] = {-1.2999999999999999e-02, 1.2999999999999999e-02, -1.2999999999999999e-02, 1.2999999999999999e-02};

/* x_dot:(z[9],x[13],u[4])->(x_dot[13]) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real *w0=w+9, w1, *w2=w+13, w3, *w4=w+18, w5, w6, w7, *w8=w+24, *w9=w+28, *w10=w+32, *w11=w+36, *w12=w+40, *w13=w+56, *w14=w+72, *w15=w+75, w16, w17, w18, *w19=w+81, *w20=w+84, *w21=w+87, *w22=w+90, *w23=w+99, *w24=w+108, *w25=w+135, *w26=w+216, *w27=w+333, *w28=w+346;
  /* #0: @0 = input[1][2] */
  casadi_copy(arg[1] ? arg[1]+7 : 0, 3, w0);
  /* #1: output[0][0] = @0 */
  casadi_copy(w0, 3, res[0]);
  /* #2: @1 = 0.5 */
  w1 = 5.0000000000000000e-01;
  /* #3: @2 = zeros(4x1) */
  casadi_clear(w2, 4);
  /* #4: @3 = 0 */
  w3 = 0.;
  /* #5: @4 = input[1][3] */
  casadi_copy(arg[1] ? arg[1]+10 : 0, 3, w4);
  /* #6: @5 = @4[0] */
  for (rr=(&w5), ss=w4+0; ss!=w4+1; ss+=1) *rr++ = *ss;
  /* #7: @5 = (-@5) */
  w5 = (- w5 );
  /* #8: @6 = @4[1] */
  for (rr=(&w6), ss=w4+1; ss!=w4+2; ss+=1) *rr++ = *ss;
  /* #9: @6 = (-@6) */
  w6 = (- w6 );
  /* #10: @7 = @4[2] */
  for (rr=(&w7), ss=w4+2; ss!=w4+3; ss+=1) *rr++ = *ss;
  /* #11: @7 = (-@7) */
  w7 = (- w7 );
  /* #12: @8 = horzcat(@3, @5, @6, @7) */
  rr=w8;
  *rr++ = w3;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  /* #13: @8 = @8' */
  /* #14: @3 = @4[0] */
  for (rr=(&w3), ss=w4+0; ss!=w4+1; ss+=1) *rr++ = *ss;
  /* #15: @5 = 0 */
  w5 = 0.;
  /* #16: @6 = @4[2] */
  for (rr=(&w6), ss=w4+2; ss!=w4+3; ss+=1) *rr++ = *ss;
  /* #17: @7 = @4[1] */
  for (rr=(&w7), ss=w4+1; ss!=w4+2; ss+=1) *rr++ = *ss;
  /* #18: @7 = (-@7) */
  w7 = (- w7 );
  /* #19: @9 = horzcat(@3, @5, @6, @7) */
  rr=w9;
  *rr++ = w3;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  /* #20: @9 = @9' */
  /* #21: @3 = @4[1] */
  for (rr=(&w3), ss=w4+1; ss!=w4+2; ss+=1) *rr++ = *ss;
  /* #22: @5 = @4[2] */
  for (rr=(&w5), ss=w4+2; ss!=w4+3; ss+=1) *rr++ = *ss;
  /* #23: @5 = (-@5) */
  w5 = (- w5 );
  /* #24: @6 = 0 */
  w6 = 0.;
  /* #25: @7 = @4[0] */
  for (rr=(&w7), ss=w4+0; ss!=w4+1; ss+=1) *rr++ = *ss;
  /* #26: @10 = horzcat(@3, @5, @6, @7) */
  rr=w10;
  *rr++ = w3;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  /* #27: @10 = @10' */
  /* #28: @3 = @4[2] */
  for (rr=(&w3), ss=w4+2; ss!=w4+3; ss+=1) *rr++ = *ss;
  /* #29: @5 = @4[1] */
  for (rr=(&w5), ss=w4+1; ss!=w4+2; ss+=1) *rr++ = *ss;
  /* #30: @6 = @4[0] */
  for (rr=(&w6), ss=w4+0; ss!=w4+1; ss+=1) *rr++ = *ss;
  /* #31: @6 = (-@6) */
  w6 = (- w6 );
  /* #32: @7 = 0 */
  w7 = 0.;
  /* #33: @11 = horzcat(@3, @5, @6, @7) */
  rr=w11;
  *rr++ = w3;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  /* #34: @11 = @11' */
  /* #35: @12 = horzcat(@8, @9, @10, @11) */
  rr=w12;
  for (i=0, cs=w8; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w9; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w10; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w11; i<4; ++i) *rr++ = *cs++;
  /* #36: @13 = @12' */
  for (i=0, rr=w13, cs=w12; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #37: @8 = input[1][1] */
  casadi_copy(arg[1] ? arg[1]+3 : 0, 4, w8);
  /* #38: @2 = mac(@13,@8,@2) */
  for (i=0, rr=w2; i<1; ++i) for (j=0; j<4; ++j, ++rr) for (k=0, ss=w13+j, tt=w8+i*4; k<4; ++k) *rr += ss[k*4]**tt++;
  /* #39: @2 = (@1*@2) */
  for (i=0, rr=w2, cs=w2; i<4; ++i) (*rr++)  = (w1*(*cs++));
  /* #40: output[0][1] = @2 */
  if (res[0]) casadi_copy(w2, 4, res[0]+3);
  /* #41: @14 = [0, 0, -9.81] */
  casadi_copy(casadi_c0, 3, w14);
  /* #42: @15 = zeros(3x1) */
  casadi_clear(w15, 3);
  /* #43: @1 = 1 */
  w1 = 1.;
  /* #44: @3 = @8[2] */
  for (rr=(&w3), ss=w8+2; ss!=w8+3; ss+=1) *rr++ = *ss;
  /* #45: @5 = sq(@3) */
  w5 = casadi_sq( w3 );
  /* #46: @6 = @8[3] */
  for (rr=(&w6), ss=w8+3; ss!=w8+4; ss+=1) *rr++ = *ss;
  /* #47: @7 = sq(@6) */
  w7 = casadi_sq( w6 );
  /* #48: @5 = (@5+@7) */
  w5 += w7;
  /* #49: @5 = (2.*@5) */
  w5 = (2.* w5 );
  /* #50: @1 = (@1-@5) */
  w1 -= w5;
  /* #51: @5 = @8[1] */
  for (rr=(&w5), ss=w8+1; ss!=w8+2; ss+=1) *rr++ = *ss;
  /* #52: @7 = (@5*@3) */
  w7  = (w5*w3);
  /* #53: @16 = @8[0] */
  for (rr=(&w16), ss=w8+0; ss!=w8+1; ss+=1) *rr++ = *ss;
  /* #54: @17 = (@16*@6) */
  w17  = (w16*w6);
  /* #55: @7 = (@7-@17) */
  w7 -= w17;
  /* #56: @7 = (2.*@7) */
  w7 = (2.* w7 );
  /* #57: @17 = (@5*@6) */
  w17  = (w5*w6);
  /* #58: @18 = (@16*@3) */
  w18  = (w16*w3);
  /* #59: @17 = (@17+@18) */
  w17 += w18;
  /* #60: @17 = (2.*@17) */
  w17 = (2.* w17 );
  /* #61: @19 = horzcat(@1, @7, @17) */
  rr=w19;
  *rr++ = w1;
  *rr++ = w7;
  *rr++ = w17;
  /* #62: @19 = @19' */
  /* #63: @1 = (@5*@3) */
  w1  = (w5*w3);
  /* #64: @7 = (@16*@6) */
  w7  = (w16*w6);
  /* #65: @1 = (@1+@7) */
  w1 += w7;
  /* #66: @1 = (2.*@1) */
  w1 = (2.* w1 );
  /* #67: @7 = 1 */
  w7 = 1.;
  /* #68: @17 = sq(@5) */
  w17 = casadi_sq( w5 );
  /* #69: @18 = sq(@6) */
  w18 = casadi_sq( w6 );
  /* #70: @17 = (@17+@18) */
  w17 += w18;
  /* #71: @17 = (2.*@17) */
  w17 = (2.* w17 );
  /* #72: @7 = (@7-@17) */
  w7 -= w17;
  /* #73: @17 = (@3*@6) */
  w17  = (w3*w6);
  /* #74: @18 = (@16*@5) */
  w18  = (w16*w5);
  /* #75: @17 = (@17-@18) */
  w17 -= w18;
  /* #76: @17 = (2.*@17) */
  w17 = (2.* w17 );
  /* #77: @20 = horzcat(@1, @7, @17) */
  rr=w20;
  *rr++ = w1;
  *rr++ = w7;
  *rr++ = w17;
  /* #78: @20 = @20' */
  /* #79: @1 = (@5*@6) */
  w1  = (w5*w6);
  /* #80: @7 = (@16*@3) */
  w7  = (w16*w3);
  /* #81: @1 = (@1-@7) */
  w1 -= w7;
  /* #82: @1 = (2.*@1) */
  w1 = (2.* w1 );
  /* #83: @6 = (@3*@6) */
  w6  = (w3*w6);
  /* #84: @16 = (@16*@5) */
  w16 *= w5;
  /* #85: @6 = (@6+@16) */
  w6 += w16;
  /* #86: @6 = (2.*@6) */
  w6 = (2.* w6 );
  /* #87: @16 = 1 */
  w16 = 1.;
  /* #88: @5 = sq(@5) */
  w5 = casadi_sq( w5 );
  /* #89: @3 = sq(@3) */
  w3 = casadi_sq( w3 );
  /* #90: @5 = (@5+@3) */
  w5 += w3;
  /* #91: @5 = (2.*@5) */
  w5 = (2.* w5 );
  /* #92: @16 = (@16-@5) */
  w16 -= w5;
  /* #93: @21 = horzcat(@1, @6, @16) */
  rr=w21;
  *rr++ = w1;
  *rr++ = w6;
  *rr++ = w16;
  /* #94: @21 = @21' */
  /* #95: @22 = horzcat(@19, @20, @21) */
  rr=w22;
  for (i=0, cs=w19; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w20; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w21; i<3; ++i) *rr++ = *cs++;
  /* #96: @23 = @22' */
  for (i=0, rr=w23, cs=w22; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #97: @1 = 0 */
  w1 = 0.;
  /* #98: @6 = 0 */
  w6 = 0.;
  /* #99: @16 = 10 */
  w16 = 10.;
  /* #100: @5 = input[2][0] */
  w5 = arg[2] ? arg[2][0] : 0;
  /* #101: @3 = input[2][1] */
  w3 = arg[2] ? arg[2][1] : 0;
  /* #102: @7 = input[2][2] */
  w7 = arg[2] ? arg[2][2] : 0;
  /* #103: @17 = input[2][3] */
  w17 = arg[2] ? arg[2][3] : 0;
  /* #104: @2 = vertcat(@5, @3, @7, @17) */
  rr=w2;
  *rr++ = w5;
  *rr++ = w3;
  *rr++ = w7;
  *rr++ = w17;
  /* #105: @9 = (@16*@2) */
  for (i=0, rr=w9, cs=w2; i<4; ++i) (*rr++)  = (w16*(*cs++));
  /* #106: @16 = @9[0] */
  for (rr=(&w16), ss=w9+0; ss!=w9+1; ss+=1) *rr++ = *ss;
  /* #107: @5 = @9[1] */
  for (rr=(&w5), ss=w9+1; ss!=w9+2; ss+=1) *rr++ = *ss;
  /* #108: @16 = (@16+@5) */
  w16 += w5;
  /* #109: @5 = @9[2] */
  for (rr=(&w5), ss=w9+2; ss!=w9+3; ss+=1) *rr++ = *ss;
  /* #110: @16 = (@16+@5) */
  w16 += w5;
  /* #111: @5 = @9[3] */
  for (rr=(&w5), ss=w9+3; ss!=w9+4; ss+=1) *rr++ = *ss;
  /* #112: @16 = (@16+@5) */
  w16 += w5;
  /* #113: @19 = vertcat(@1, @6, @16) */
  rr=w19;
  *rr++ = w1;
  *rr++ = w6;
  *rr++ = w16;
  /* #114: @15 = mac(@23,@19,@15) */
  for (i=0, rr=w15; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w23+j, tt=w19+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #115: @14 = (@14+@15) */
  for (i=0, rr=w14, cs=w15; i<3; ++i) (*rr++) += (*cs++);
  /* #116: @15 = zeros(3x1) */
  casadi_clear(w15, 3);
  /* #117: @24 = 
  [[0, 0, 0, 1, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 1, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 1, 0, 0, 0]] */
  casadi_copy(casadi_c1, 27, w24);
  /* #118: @23 = zeros(9x1) */
  casadi_clear(w23, 9);
  /* #119: @25 = 
  [[0.333333, 0.333333, 0.333333, 14.1801, -3.365, 348.336, 2.22295e+07, -3.86099e+07, 2.40539e+07], 
   [0.333333, 0.333333, 0.333333, 14.1801, -3.365, 348.336, 2.22295e+07, -3.86099e+07, 2.40539e+07], 
   [0.333333, 0.333333, 0.333333, 14.1801, -3.365, 348.336, 2.22295e+07, -3.86099e+07, 2.40539e+07], 
   [2.20268e-13, 2.20268e-13, 2.20268e-13, 3982.34, -115.139, 524.703, -4.69428e+06, 635016, -3.08287e+07], 
   [3.41061e-13, 3.41061e-13, 3.41061e-13, 105.346, 3965.38, 1318.29, -3.03627e+07, 3.32793e+07, -5.88164e+07], 
   [-9.23706e-13, -9.23706e-13, -9.23706e-13, 19.6526, -5.26986, -656.972, 6.74968e+07, -7.29096e+07, 1.40624e+08], 
   [-4.66294e-15, -4.66294e-15, -4.66294e-15, -0.0650886, 0.142748, -25.5631, 4.64635e+06, -662768, 1.23404e+06], 
   [-6.21725e-15, -6.21725e-15, -6.21725e-15, -0.0666153, 0.120934, -25.5663, 656486, 3.32688e+06, 1.23279e+06], 
   [5.27356e-16, 5.27356e-16, 5.27356e-16, -0.00232302, 0.0206963, 1.83619, 116921, -123370, 3.86161e+06]] */
  casadi_copy(casadi_c2, 81, w25);
  /* #120: @22 = input[0][0] */
  casadi_copy(arg[0], 9, w22);
  /* #121: @23 = mac(@25,@22,@23) */
  for (i=0, rr=w23; i<1; ++i) for (j=0; j<9; ++j, ++rr) for (k=0, ss=w25+j, tt=w22+i*9; k<9; ++k) *rr += ss[k*9]**tt++;
  /* #122: @22 = zeros(9x1) */
  casadi_clear(w22, 9);
  /* #123: @26 = 
  [[-7.88258e-15, -7.54952e-15, -2.5413e-13, -3.69482e-13, 8.34888e-14, -1.06581e-14, 3.64153e-13, -1.95399e-13, -3.12639e-13, 1.15108e-12, -5.26662e-15, 3.88578e-15, 2.62901e-13], 
   [-7.88258e-15, -7.54952e-15, -2.5413e-13, -3.69482e-13, 8.34888e-14, -1.06581e-14, 3.64153e-13, -1.95399e-13, -3.12639e-13, 1.15108e-12, -5.26662e-15, 3.88578e-15, 2.62901e-13], 
   [-7.88258e-15, -7.54952e-15, -2.5413e-13, -3.69482e-13, 8.34888e-14, -1.06581e-14, 3.64153e-13, -1.95399e-13, -3.12639e-13, 1.15108e-12, -5.26662e-15, 3.88578e-15, 2.62901e-13], 
   [-0.245685, -0.302625, -0.995278, -37.619, 61.8649, 25.9943, -0.460646, -1983.1, -105.092, -19.3224, -5.56061, 9.21016, 6.96015], 
   [0.555995, 0.187681, -0.44324, 9.61191, -26.0811, 54.4413, 0.590975, 114.38, -1965.34, 4.99228, -8.96158, -5.4324, -15.6929], 
   [-21.0874, -37.4612, -943.674, -701.679, 328.85, -166.29, 792.079, -481.086, -1324.13, 2321.98, -7.48582, 6.21534, 511.973], 
   [508175, 825820, -39902.6, -5.33165e+07, -8.07468e+06, 7.26623e+06, -6.64216e+06, 3.74143e+06, 3.02016e+07, -6.58976e+07, -344978, 209533, -2.23173e+07], 
   [-461476, -886518, 83317.1, 1.0134e+08, 8.73044e+06, -5.9467e+06, -923606, 335161, -3.30519e+07, 7.11825e+07, 390605, -234965, 3.10329e+07], 
   [1.09005e+06, 1.74383e+06, 7.06138e+06, -5.52808e+07, -1.7413e+07, 3.60253e+06, -2.38085e+07, 2.90742e+07, 5.94623e+07, -1.35032e+08, 561818, -318603, -4.22131e+07]] */
  casadi_copy(casadi_c3, 117, w26);
  /* #124: @19 = input[1][0] */
  casadi_copy(arg[1], 3, w19);
  /* #125: @27 = vertcat(@19, @8, @0, @4) */
  rr=w27;
  for (i=0, cs=w19; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w8; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w0; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w4; i<3; ++i) *rr++ = *cs++;
  /* #126: @22 = mac(@26,@27,@22) */
  for (i=0, rr=w22; i<1; ++i) for (j=0; j<9; ++j, ++rr) for (k=0, ss=w26+j, tt=w27+i*13; k<13; ++k) *rr += ss[k*9]**tt++;
  /* #127: @23 = (@23+@22) */
  for (i=0, rr=w23, cs=w22; i<9; ++i) (*rr++) += (*cs++);
  /* #128: @22 = zeros(9x1) */
  casadi_clear(w22, 9);
  /* #129: @28 = 
  [[4.26326e-13, -3.71259e-13, 5.22249e-13, 1.13687e-13], 
   [4.26326e-13, -3.71259e-13, 5.22249e-13, 1.13687e-13], 
   [4.26326e-13, -3.71259e-13, 5.22249e-13, 1.13687e-13], 
   [-4.86009, 36.9392, 13.0222, -61.3992], 
   [-33.0226, 16.4525, 36.822, -16.6668], 
   [2109.5, -1085.75, 917.306, 441.25], 
   [-6.46974e+07, 7.63558e+06, -1.01227e+07, 1.40004e+07], 
   [4.99564e+07, 1.20931e+07, -9.51857e+06, 4.96026e+06], 
   [-6.9073e+07, 4.83939e+07, -5.24095e+07, -2.10625e+07]] */
  casadi_copy(casadi_c4, 36, w28);
  /* #130: @22 = mac(@28,@2,@22) */
  for (i=0, rr=w22; i<1; ++i) for (j=0; j<9; ++j, ++rr) for (k=0, ss=w28+j, tt=w2+i*4; k<4; ++k) *rr += ss[k*9]**tt++;
  /* #131: @23 = (@23+@22) */
  for (i=0, rr=w23, cs=w22; i<9; ++i) (*rr++) += (*cs++);
  /* #132: @15 = mac(@24,@23,@15) */
  for (i=0, rr=w15; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w24+j, tt=w23+i*9; k<9; ++k) *rr += ss[k*3]**tt++;
  /* #133: @14 = (@14+@15) */
  for (i=0, rr=w14, cs=w15; i<3; ++i) (*rr++) += (*cs++);
  /* #134: output[0][2] = @14 */
  if (res[0]) casadi_copy(w14, 3, res[0]+7);
  /* #135: @1 = 0 */
  w1 = 0.;
  /* #136: @6 = 10 */
  w6 = 10.;
  /* #137: @2 = (@6*@2) */
  for (i=0, rr=w2, cs=w2; i<4; ++i) (*rr++)  = (w6*(*cs++));
  /* #138: @8 = @2' */
  casadi_copy(w2, 4, w8);
  /* #139: @9 = [0, 0.235, 0, -0.235] */
  casadi_copy(casadi_c5, 4, w9);
  /* #140: @1 = mac(@8,@9,@1) */
  for (i=0, rr=(&w1); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w8+j, tt=w9+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #141: @6 = -0.03 */
  w6 = -2.9999999999999999e-02;
  /* #142: @16 = @4[1] */
  for (rr=(&w16), ss=w4+1; ss!=w4+2; ss+=1) *rr++ = *ss;
  /* #143: @6 = (@6*@16) */
  w6 *= w16;
  /* #144: @16 = @4[2] */
  for (rr=(&w16), ss=w4+2; ss!=w4+3; ss+=1) *rr++ = *ss;
  /* #145: @6 = (@6*@16) */
  w6 *= w16;
  /* #146: @1 = (@1+@6) */
  w1 += w6;
  /* #147: @6 = 0.03 */
  w6 = 2.9999999999999999e-02;
  /* #148: @1 = (@1/@6) */
  w1 /= w6;
  /* #149: output[0][3] = @1 */
  if (res[0]) res[0][10] = w1;
  /* #150: @1 = 0.03 */
  w1 = 2.9999999999999999e-02;
  /* #151: @6 = @4[2] */
  for (rr=(&w6), ss=w4+2; ss!=w4+3; ss+=1) *rr++ = *ss;
  /* #152: @1 = (@1*@6) */
  w1 *= w6;
  /* #153: @6 = @4[0] */
  for (rr=(&w6), ss=w4+0; ss!=w4+1; ss+=1) *rr++ = *ss;
  /* #154: @1 = (@1*@6) */
  w1 *= w6;
  /* #155: @6 = 0 */
  w6 = 0.;
  /* #156: @8 = @2' */
  casadi_copy(w2, 4, w8);
  /* #157: @9 = [0.235, 0, -0.235, 0] */
  casadi_copy(casadi_c6, 4, w9);
  /* #158: @6 = mac(@8,@9,@6) */
  for (i=0, rr=(&w6); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w8+j, tt=w9+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #159: @1 = (@1-@6) */
  w1 -= w6;
  /* #160: @6 = 0.03 */
  w6 = 2.9999999999999999e-02;
  /* #161: @1 = (@1/@6) */
  w1 /= w6;
  /* #162: output[0][4] = @1 */
  if (res[0]) res[0][11] = w1;
  /* #163: @1 = 0 */
  w1 = 0.;
  /* #164: @2 = @2' */
  /* #165: @8 = [-0.013, 0.013, -0.013, 0.013] */
  casadi_copy(casadi_c7, 4, w8);
  /* #166: @1 = mac(@2,@8,@1) */
  for (i=0, rr=(&w1); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w2+j, tt=w8+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #167: @6 = 0.06 */
  w6 = 5.9999999999999998e-02;
  /* #168: @1 = (@1/@6) */
  w1 /= w6;
  /* #169: output[0][5] = @1 */
  if (res[0]) res[0][12] = w1;
  return 0;
}

/* default_expl_ode_fun:(i0[13],i1[4],i2[])->(o0[13]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i;
  casadi_real **res1=res+1, *rr;
  const casadi_real **arg1=arg+3, *cs;
  casadi_real *w0=w+382, *w1=w+391, *w2=w+394, *w3=w+398, *w4=w+401, *w5=w+404, w6, w7, w8, w9, *w10=w+421;
  /* #0: @0 = zeros(9x1) */
  casadi_clear(w0, 9);
  /* #1: @1 = input[0][0] */
  casadi_copy(arg[0], 3, w1);
  /* #2: @2 = input[0][1] */
  casadi_copy(arg[0] ? arg[0]+3 : 0, 4, w2);
  /* #3: @3 = input[0][2] */
  casadi_copy(arg[0] ? arg[0]+7 : 0, 3, w3);
  /* #4: @4 = input[0][3] */
  casadi_copy(arg[0] ? arg[0]+10 : 0, 3, w4);
  /* #5: @5 = vertcat(@1, @2, @3, @4) */
  rr=w5;
  for (i=0, cs=w1; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w2; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w3; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w4; i<3; ++i) *rr++ = *cs++;
  /* #6: @6 = input[1][0] */
  w6 = arg[1] ? arg[1][0] : 0;
  /* #7: @7 = input[1][1] */
  w7 = arg[1] ? arg[1][1] : 0;
  /* #8: @8 = input[1][2] */
  w8 = arg[1] ? arg[1][2] : 0;
  /* #9: @9 = input[1][3] */
  w9 = arg[1] ? arg[1][3] : 0;
  /* #10: @2 = vertcat(@6, @7, @8, @9) */
  rr=w2;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  /* #11: @10 = x_dot(@0, @5, @2) */
  arg1[0]=w0;
  arg1[1]=w5;
  arg1[2]=w2;
  res1[0]=w10;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #12: output[0][0] = @10 */
  casadi_copy(w10, 13, res[0]);
  return 0;
}

CASADI_SYMBOL_EXPORT int default_expl_ode_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int default_expl_ode_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int default_expl_ode_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void default_expl_ode_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int default_expl_ode_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void default_expl_ode_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void default_expl_ode_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void default_expl_ode_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int default_expl_ode_fun_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int default_expl_ode_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real default_expl_ode_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* default_expl_ode_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* default_expl_ode_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* default_expl_ode_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* default_expl_ode_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int default_expl_ode_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 10;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 434;
  return 0;
}

CASADI_SYMBOL_EXPORT int default_expl_ode_fun_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 10*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 3*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 434*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
