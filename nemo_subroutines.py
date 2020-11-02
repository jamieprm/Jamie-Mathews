import numpy as np
from scipy.interpolate import griddata


# =============================================================================
# Utility functions
# =============================================================================

def expnd(A,d):
    #turning 3 type to 3 tupe tensor
    A=np.repeat(A[np.newaxis,:,:], d, axis=0)
    #A=np.delete(np.delete(A,103,1),99,2)
    return A

def dlt(e3t):
    E=np.delete(np.delete(np.delete(e3t,np.shape(e3t)[0]-1,0),np.shape(e3t)[1]-1,1),np.shape(e3t)[2]-1,2)
    return E

def null(e,d):
    E=e
    for i in range(1,d+1):
        E[np.shape(e)[0]-i,:,:]=None
        E[:,np.shape(e)[1]-i,:]=None
        E[:,:,np.shape(e)[2]-i]=None
    return E

# =============================================================================
# density functions
# =============================================================================

def calc_rho(sa,ct,p0):
    #computes in situ density using TEOS-10 (boussinesq) from Roquet et al. (2015)
    #absolute salinity sa in kg/kg, conservative temp ct in deg C, pressure p0 in dbar
    SAu = 40*35.16504/35 
    CTu = 40 
    Zu = 1e4
    deltaS = 32
    R000 = 8.0189615746e+02 
    R100 = 8.6672408165e+02 
    R200 = -1.7864682637e+03
    R300 = 2.0375295546e+03 
    R400 = -1.2849161071e+03 
    R500 = 4.3227585684e+02
    R600 = -6.0579916612e+01 
    R010 = 2.6010145068e+01 
    R110 = -6.5281885265e+01
    R210 = 8.1770425108e+01
    R310 = -5.6888046321e+01 
    R410 = 1.7681814114e+01
    R510 = -1.9193502195e+00 
    R020 = -3.7074170417e+01 
    R120 = 6.1548258127e+01
    R220 = -6.0362551501e+01
    R320 = 2.9130021253e+01 
    R420 = -5.4723692739e+00
    R030 = 2.1661789529e+01 
    R130 = -3.3449108469e+01 
    R230 = 1.9717078466e+01
    R330 = -3.1742946532e+00 
    R040 = -8.3627885467e+00 
    R140 = 1.1311538584e+01
    R240 = -5.3563304045e+00 
    R050 = 5.4048723791e-01 
    R150 = 4.8169980163e-01
    R060 = -1.9083568888e-01 
    R001 = 1.9681925209e+01 
    R101 = -4.2549998214e+01
    R201 = 5.0774768218e+01 
    R301 = -3.0938076334e+01 
    R401 = 6.6051753097e+00
    R011 = -1.3336301113e+01 
    R111 = -4.4870114575e+00
    R211 = 5.0042598061e+00
    R311 = -6.5399043664e-01
    R021 = 6.7080479603e+00
    R121 = 3.5063081279e+00
    R221 = -1.8795372996e+00 
    R031 = -2.4649669534e+00 
    R131 = -5.5077101279e-01
    R041 = 5.5927935970e-01
    R002 = 2.0660924175e+00
    R102 = -4.9527603989e+00
    R202 = 2.5019633244e+00
    R012 = 2.0564311499e+00 
    R112 = -2.1311365518e-01
    R022 = -1.2419983026e+00 
    R003 = -2.3342758797e-02 
    R103 = -1.8507636718e-02
    R013 = 3.7969820455e-01

    SA = sa
    CT = ct
    Z = -p0

    ss = np.sqrt ( (SA+deltaS)/SAu );
    tt = CT / CTu;
    zz =  -Z / Zu;
    rz3 = R013 * tt + R103 * ss + R003
    rz2 = (R022 * tt+R112 * ss+R012) * tt+(R202 * ss+R102) * ss+R002
    rz1 = (((R041 * tt+R131 * ss+R031) * tt + (R221 * ss+R121) * ss+R021) * tt + ((R311 * ss+R211) * ss+R111) * ss+R011) * tt + (((R401 * ss+R301) * ss+R201) * ss+R101) * ss+R001
    rz0 = (((((R060 * tt+R150 * ss+R050) * tt + (R240 * ss+R140) * ss+R040) * tt + ((R330 * ss+R230) * ss+R130) * ss+R030) * tt + (((R420 * ss+R320) * ss+R220) * ss+R120) * ss+R020) * tt + ((((R510 * ss+R410) * ss+R310) * ss+R210) * ss+R110) * ss+R010) * tt +(((((R600 * ss+R500) * ss+R400) * ss+R300) * ss+R200) * ss+R100) * ss+R000
    r = ( ( rz3 * zz + rz2 ) * zz + rz1 ) * zz + rz0

    Zu = 1e4 
    zz = -Z / Zu
    R00 = 4.6494977072e+01 
    R01 = -5.2099962525e+00
    R02 = 2.2601900708e-01
    R03 = 6.4326772569e-02 
    R04 = 1.5616995503e-02
    R05 = -1.7243708991e-03
    r0 = (((((R05 * zz+R04) * zz+R03 ) * zz+R02 ) * zz+R01) * zz+R00) * zz
    rho = r0 + r

    return rho

# def calc_sigma0(sa,ct):
#         # Computes sigma0 from TEOS-10 (Bousinesq)
#         # absolute salinity sa in kg/kg, conservative temp ct in deg C
#         sigma0 = calc_rho(sa,ct,0)

#         return sigma0

def calc_sigma0(sp,ti):
        # Computes sigma0 using the code /home/users/atb299/CDFTOOLS_Nov18/src/eos.f90
        zt  = ti          # in situ temp (i.e. not ct)
        zs  = sp          # in situ salinity (i.e., not sa)
        zsr = np.sqrt(zs) # square root of interpolated S

        #compute volumic mass pure water at atm pressure
        zr1 = ( ( ( ( 6.536332e-9*zt-1.120083e-6 )*zt+1.001685e-4)*zt -9.095290e-3 )*zt+6.793952e-2 )*zt+999.842594
        #seawater volumic mass atm pressure
        zr2= ( ( ( 5.3875e-9*zt-8.2467e-7 )*zt+7.6438e-5 ) *zt-4.0899e-3 ) *zt+0.824493
        zr3= ( -1.6546e-6*zt+1.0227e-4 ) *zt-5.72466e-3
        zr4= 4.8314e-4
        zrau0 = 1000
        #potential volumic mass (reference to the surface)
        sigma0 = ( zr4*zs + zr3*zsr + zr2 ) *zs + zr1 - zrau0

        return sigma0+1000
    
# =============================================================================
# #averaging functions
# =============================================================================

def av(S,d,p):
    #d=0,1,2 respectively corresponds to the x,y,z plane averaging
    #p=1 average +1/2, p=-1 average to -1/2 
    S=null(S,1)
    if d==0: #move to vw
        m=0
        n=1
    elif d==1:  #move to uw
        m=0
        n=2
    elif d==2:  #move to f
        m=1
        n=2
    S1=np.roll(S,-p,n)
    S2=np.roll(S,-p,m)
    S3=np.roll(np.roll(S,-p,n),-p,m)
    S_av=(S+S1+S2+S3)/4
    #S_av=null(S_av, 1)
    S_av=np.roll(S_av,1-m,0)
    return S_av

def av_fw(S):
    #averaging to centre of cube
    S1=np.roll(S,-1,0)  #w
    S2=np.roll(S,-1,1)  #v
    S3=np.roll(S,-1,2)  #u
    S4=np.roll(S1,-1,1) #vw
    S5=np.roll(S1,-1,2) #uw
    S6=np.roll(S2,-1,2) #f   
    S7=np.roll(S6,-1,0) #fw
    S_av=(S+S1+S2+S3+S4+S5+S6+S7)/8
    S_av=null(S_av,1)
    S_av=np.roll(S_av,1,0)
    return S_av

def interp1d(gdept,gphitp,RHO,gdepw,lo,METHOD):
    ii=np.shape(gdept)[0]
    jj=np.shape(gdept)[1]
    Ti = RHO[0:ii,0:jj,lo]
    zi = gdept[0:ii,0:jj,lo]
    yi = gphitp[0:ii,0:jj,lo]
    zf = gdepw[0:ii,0:jj,lo]
    yf = gphitp[0:ii,0:jj,lo]
    Nz, Ny = Ti.shape
    Ti_vec = np.reshape(Ti,(Nz*Ny))
    zi_vec = np.reshape(zi,(Nz*Ny))
    yi_vec = np.reshape(yi,(Nz*Ny))
    zf_vec = np.reshape(zf,(Nz*Ny))
    yf_vec = np.reshape(yf,(Nz*Ny))
    points=(zi_vec, yi_vec)
    Tf = griddata(points, Ti_vec, (zf_vec, yf_vec), method=METHOD)
    Tf=np.reshape(Tf,(Nz,Ny))
    return Tf

def interp2dhor(glamt,gphit,RHO,glamf,gphif,k,METHOD):
    ii=np.shape(gphit)[0]
    jj=np.shape(glamt)[1]
    Ti = RHO[k,0:ii,0:jj]
    xi = glamt[0:ii,0:jj]
    yi = gphit[0:ii,0:jj]
    xf = glamf[0:ii,0:jj]
    yf = gphif[0:ii,0:jj]
    Ny, Nx = Ti.shape
    Ti_vec = np.reshape(Ti,(Ny*Nx))
    xi_vec = np.reshape(xi,(Ny*Nx))
    yi_vec = np.reshape(yi,(Ny*Nx))
    xf_vec = np.reshape(xf,(Ny*Nx))
    yf_vec = np.reshape(yf,(Ny*Nx))
    Tf = griddata((xi_vec, yi_vec), Ti_vec, (xf_vec, yf_vec), method=METHOD)
    Tf=np.reshape(Tf,(Ny,Nx))
    return Tf

def interp2dvert(gdept,gphit,RHO,gdepw,gphif,lo,METHOD):
    gphitp=expnd(gphit, len(RHO))
    gphifp=expnd(gphif, len(RHO))
    ii=np.shape(RHO)[0]
    jj=np.shape(RHO)[1]
    Ti = RHO[0:ii,0:jj,lo]
    xi = gdept[0:ii,0:jj,lo]
    yi = gphitp[0:ii,0:jj,lo]
    xf = gdepw[0:ii,0:jj,lo]
    yf = gphifp[0:ii,0:jj,lo]
    Ny, Nx = Ti.shape
    Ti_vec = np.reshape(Ti,(Ny*Nx))
    xi_vec = np.reshape(xi,(Ny*Nx))
    yi_vec = np.reshape(yi,(Ny*Nx))
    xf_vec = np.reshape(xf,(Ny*Nx))
    yf_vec = np.reshape(yf,(Ny*Nx))
    Tf = griddata((xi_vec, yi_vec), Ti_vec, (xf_vec, yf_vec), method=METHOD)
    Tf=np.reshape(Tf,(Ny,Nx))
    return Tf

    
def interp3dij(glamt,gphit,RHO,glamfw,gphifw, METHOD):
    glamtp=expnd(glamt, len(RHO))
    gphitp=expnd(gphit, len(RHO))
    glamfwp=expnd(glamfw, len(RHO))
    gphifwp=expnd(gphifw, len(RHO))
    size=np.size(RHO)
    xi=np.reshape(glamtp,(size))
    yi=np.reshape(gphitp,(size))
    xf=np.reshape(glamfwp,(size))
    yf=np.reshape(gphifwp,(size))
    
    initial=np.zeros((size,2))
    initial[:,0]=xi[:]
    initial[:,1]=yi[:]
    
    final=np.zeros((size,2))
    final[:,0]=xf[:]
    final[:,1]=yf[:]
    
    values=np.reshape(RHO,(size))
    
    new=griddata(initial, values, final)
    new=np.reshape(new,np.shape(RHO))
    return new, initial, values, xi

# def interp3dk(gdept,gphit,RHO,gdepw,gphif, METHOD):
#     gphitp=expnd(gphit, len(RHO))
#     gphifp=expnd(gphif, len(RHO))
#     ii=np.shape(RHO)[0]
#     jj=np.shape(RHO)[1]
#     Nz, Ny, Nx = np.shape(RHO)
#     size=Nz*Ny
#     new=np.zeros(np.shape(RHO))
    
#     for lo in range(0,np.shape(RHO)[2]-1):

#         Ti = RHO[0:ii,0:jj,lo]
#         zi = gdept[0:ii,0:jj,lo]
#         yi = gphitp[0:ii,0:jj,lo]
#         zf = gdepw[0:ii,0:jj,lo]
#         yf = gphifp[0:ii,0:jj,lo]
#         Ti_vec = np.reshape(Ti,(size))
#         zi_vec = np.reshape(zi,(size))
#         yi_vec = np.reshape(yi,(size))
#         zf_vec = np.reshape(zf,(size))
#         yf_vec = np.reshape(yf,(size))
#         Tf = griddata((zi_vec, yi_vec), Ti_vec, (zf_vec, yf_vec), method=METHOD)
#         new[:,:,lo]=np.reshape(Tf,(Nz,Ny))
#     return new

def interp3dk(gdept,gphit,RHO,gdepw,gphif,METHOD):
    Tf=np.zeros(np.shape(RHO))
    for lo in  range(0,np.shape(RHO)[2]-1):
        Tf[:,:,lo]=interp2dvert(gdept, gphit, RHO, gdepw, gphif, lo, METHOD)
    
    return Tf

def interp3dlatlong(glamt,gphit,RHO,glamf,gphif,METHOD):
    Tf=np.zeros(np.shape(RHO))
    for k in  range(0,len(RHO)-1):
        Tf[k,:,:]=interp2dhor(glamt, gphit, RHO, glamf, gphif, k, METHOD)
    
    return Tf

def interp4d(glamtp,gphitp,gdept,RHO,glamfwp,gphifwp,gdepfw, METHOD):
    size=np.size(gdept)
    xi=np.reshape(glamtp,(size))
    yi=np.reshape(gphitp,(size))
    zi=np.reshape(gdept,(size))
    xf=np.reshape(glamfwp,(size))
    yf=np.reshape(gphifwp,(size))
    zf=np.reshape(gdepfw,(size))
    
    initial=np.zeros((size,3))
    initial[:,0]=xi[:]
    initial[:,1]=yi[:]
    initial[:,2]=zi[:]
    
    final=np.zeros((size,3))
    final[:,0]=xf[:]
    final[:,1]=yf[:]
    final[:,2]=zf[:]
    
    values=np.reshape(RHO,(size))
    
    new=griddata(initial, values, final)
    new=np.reshape(new,np.shape(RHO))
    return new


# =============================================================================
# vorticity Functions
# =============================================================================

#calculating the coriolis parameter 
    
def cor(gphi):
    #given in deg
    omega=7.2921159e-5
    gphi=np.radians(gphi)
    return [2*omega*np.sin(gphi),2*omega*np.cos(gphi)]


#Calculating all vorticities. 
#When followed by a greek index it follows an itterative proccess, else matrix notation
           
           
def calc_vortyz(U,V,e1u,e2v,e1f,e2f):
    dA=e1f*e2f
    eU=U*e1u
    eV=V*e2v
    du=np.roll(eU,-1,1)-eU
    dv=np.roll(eV,-1,2)-eV
    vort=(dv - du)/dA       #defined at f points 
    
    vort=null(vort,1)
    return vort

def calc_vortyy(U,W,e1uw,e3uw):
    du=-np.roll(U,-1,0)+U
    dw=np.roll(W,-1,2)-W
    vort=du/np.roll(e3uw,-1,0)-np.roll(dw/e1uw,-1,0)
    
    vort=null(vort,1)
    vort=np.roll(vort,1,0)   #bring back to uw(0) point
    return vort

    
def calc_vortyx(V,W,e2vw,e3vw):
    dv=-np.roll(V,-1,0)+V
    dw=np.roll(W,-1,1)-W
    vort=np.roll(dw/e2vw,-1,0)-dv/np.roll(e3vw,-1,0)
    
    vort=null(vort,1)
    vort=np.roll(vort,1,0)  #bring back to uw(0)
    return vort


#Calculating the potential vorticity


def calc_PVf(f,e3fw,RHO,SIG):
    #averaging to the sides and centre of the cube
    SIGk=av(SIG,2)
    RHO=av_fw(RHO)
    #sides vorticity and SIGMA
    Sk=SIGk
    dSk=Sk-np.roll(Sk,-1,0)
    PV=-f*dSk/(RHO*e3fw)
    return PV

def calc_PV1(gvori,gvorj,gvork,e1vw,e2uw,e3f,RHO,SIG):
    #averaging everything over to the correct positions
    #SIGfw=av_fw(SIG)
    SIGfw=SIG       #remove later
    RHOi=av(RHO,0,1)
    RHOj=av(RHO,1,1)
    RHOk=av(RHO,2,1)
    # RHOi=RHO
    # RHOj=RHO
    # RHOk=RHO
    #calculating the divergence
    dSi=-(np.roll(SIGfw,+1,2)-SIGfw)/e1vw
    dSj=-(np.roll(SIGfw,+1,1)-SIGfw)/e2uw
    dSk=(-np.roll(SIGfw,-1,0)+SIGfw)/e3f
    #PV formula
    PVi=-gvori*dSi/RHOi
    PVj=-gvorj*dSj/RHOj
    PVk=-gvork*dSk/RHOk
    #averaging over to fw 
    PVi=(np.roll(PVi,-1,2)+PVi)/2
    PVj=(np.roll(PVj,-1,1)+PVj)/2
    #PVk=(np.roll(PVk,+1,0)+PVk)/2
    PVi=null(PVi, 1)
    PVj=null(PVj, 1)
    PVk=null(PVk, 1)
    
    PV=PVk+PVj+PVi
    return [PV, PVi, PVj, PVk]

def calc_PV(gvori,gvorj,gvork,e1vw,e2uw,e3f,gdept,glamu,glamv,gdepw,glamf,gphiu,gphiv,gphif,RHO,SIGfw):
    #Interpolating everything over to the correct positions
    SIG=SIGfw
    RHOi=RHO
    RHOj=RHO
    RHOk=RHO
    #calculating the divergence
    dSi=-(np.roll(SIG,+1,2)-SIG)/e1vw
    dSj=-(np.roll(SIG,+1,1)-SIG)/e2uw
    dSk=(-np.roll(SIG,-1,0)+SIG)/e3f
    #PV formula
    PVi=-gvori*dSi/RHOi
    PVj=-gvorj*dSj/RHOj
    PVk=-gvork*dSk/RHOk
    PVk=null(PVk,2)
    PVj=null(PVj,1)
    #Interpolating over to fw 
    PVi=interp3dlatlong(glamv, gphiv, PVi, glamf, gphif, 'linear')
    PVj=interp3dlatlong(glamu, gphiu, PVj, glamf, gphif, 'linear')
    PVk=interp3dk(gdept, gphiu, PVk, gdepw, gphiu, 'linear')
    #adding up total PV
    PV=PVk+PVj+PVi
    return [PV, PVi, PVj, PVk]

def calc_PVt(gvori,gvorj,gvork,e1vw,e2uw,e3f,RHO,SIG):#,glamf,glamt,gphif,gphit):
    #averaging everything over to the correct positions
    SIGfw=av_fw(SIG)
    RHOi=av(RHO,0,1)
    RHOj=av(RHO,1,1)
    RHOk=av(RHO,2,1)
    #calculating the divergence
    dSi=-(np.roll(SIGfw,+1,2)-SIGfw)/e1vw
    dSj=-(np.roll(SIGfw,+1,1)-SIGfw)/e2uw
    dSk=(-np.roll(SIGfw,-1,0)+SIGfw)/e3f
    #PV formula
    PVi=-gvori*dSi#/RHOi
    PVj=-gvorj*dSj#/RHOj
    PVk=-gvork*dSk#/RHOk
    #averaging over to t
    PVi=av(PVi,0,-1)/RHO
    PVj=av(PVj,1,-1)/RHO
    PVk=av(PVk,2,-1)/RHO
    #PVk=interp3dlatlong(glamf, gphif, PVk, glamt, gphit, 'linear')
    # PVi=null(PVi, 1)
    # PVj=null(PVj, 1)
    # PVk=null(PVk, 1)
        
    PV=PVk+PVj+PVi
    return [PV, PVi, PVj, PVk]
 
# =============================================================================
# conservation functions
# =============================================================================

def divergence(u,v,w,e1t,e2t,e3t,e2u,e1v):
    u=e2t*u
    v=e1t*v
    w=w
    A=e1t*e2t
    du=np.roll(u,-1,2)-u
    dv=np.roll(v,-1,1)-v
    dw=-np.roll(w,-1,0)+w
    return (du+dv)/A+dw/(e3t)

def mat_div(PV1,PV2,U,V,W,T,e1,e2,e3):
    dt=(PV2-PV1)/T
    PV0=(PV1+PV2)/2
    div=U*(np.roll(PV0,-1,2)-PV0)/e1+V*(np.roll(PV0,-1,1)-PV0)/e2+W*(PV0-np.roll(PV0,-1,0))/e3
    return val




            
    


