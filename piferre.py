#!/home/callende/anaconda3/bin/python3
'''
Interface to use FERRE from python for DESI/BOSS/WEAVE data

use: piferre path-to-spectra [truthfile]

e.g. piferre 
/home/callende/work/desi/berkeley/spectro/redux/dc17a2/spectra-64 

Author C. Allende Prieto
'''
import pdb
import sys
import os
import glob
import re
from numpy import arange,loadtxt,savetxt,zeros,ones,nan,sqrt,interp,concatenate,array,reshape,min,max,where,divide,mean, stack
from astropy.io import fits
import astropy.table as tbl
import astropy.units as units
import matplotlib.pyplot as plt
import subprocess
import datetime, time

#extract the header of a synthfile
def head_synth(synthfile):
    file=open(synthfile,'r')
    line=file.readline()
    header={}
    while (1):
        line=file.readline()
        part=line.split('=')
        if (len(part) < 2): break
        k=part[0].strip()
        v=part[1].strip()
        header[k]=v
    return header

#extract the wavelength array for a FERRE synth file
def lambda_synth(synthfile):
    header=head_synth(synthfile)
    tmp=header['WAVE'].split()
    npix=int(header['NPIX'])
    step=float(tmp[1])
    x0=float(tmp[0])
    x=arange(npix)*step+x0
    if header['LOGW']:
      if int(header['LOGW']) == 1: x=10.**x
      if int(header['LOGW']) == 2: x=exp(x)  
    return x

#create a slurm script for a given pixel
def write_slurm(pixel,nthreads=1,path=None,ngrids=None, suffix='', pre='n'):
    ferre=os.environ['HOME']+"/ferre/src/a.out"
    python_path=os.environ['HOME']+"/piferre"
    try: 
      host=os.environ['HOST']
    except:
      host='Unknown'

    now=time.strftime("%c")
    sdir=pixel[:-2]
    if path is None: path='.'
    if ngrids is None: ngrids=1

    f=open(os.path.join(path,pixel+suffix+'.slurm'),'w')
    f.write("#!/bin/bash \n")
    f.write("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# \n")
    f.write("#This script was written by piferre.py on "+now+" \n") 
    if host[:4] == 'cori':
      f.write("#SBATCH --qos=regular" + "\n")
      f.write("#SBATCH --constraint=haswell" + "\n")
      f.write("#SBATCH --time=180"+"\n") #minutes
      f.write("#SBATCH --ntasks=1" + "\n")
      f.write("#SBATCH --cpus-per-task="+str(nthreads*2)+"\n")
    else:
      f.write("#SBATCH  -J "+str(pixel)+" \n")
      f.write("#SBATCH  -p batch"+" \n")
      f.write("#SBATCH  -o "+str(pixel)+"_%j.out"+" \n")
      f.write("#SBATCH  -e "+str(pixel)+"_%j.err"+" \n")
      f.write("#SBATCH  -n "+str(nthreads)+" \n")
      f.write("#SBATCH  -t 04:00:00"+" \n") #hh:mm:ss
      f.write("#SBATCH  -D "+os.path.abspath(path)+" \n")
    f.write("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# \n")
    f.write("export OMP_NUM_THREADS="+str(nthreads)+"\n")
    f.write("cd "+os.path.abspath(path)+"\n")
    for i in range(ngrids):
      #f.write("cp input.nml"+suffix+"_"+str(i)+" input.nml \n")
      f.write("time "+ferre+" input.nml"+suffix+"_"+str(i)+" >& log_"+str(i))
      #if (i == 8): 
      #  f.write( "  \n")
      #else:
      #  f.write( " & \n")
      f.write( " & \n")
    if ngrids > 1:
      f.write("wait \n")
      f.write("python3 -c \"import sys; sys.path.insert(0, '"+python_path+ \
              "'); from piferre import opfmerge, write_tab_fits, write_mod_fits; opfmerge(\'"+\
              str(pixel)+suffix+"\',pre='"+pre+"\'); write_tab_fits(\'"+\
              str(pixel)+suffix+"\',pre='"+pre+"\'); write_mod_fits(\'"+\
              str(pixel)+suffix+"\')\"\n")
    f.close()
    os.chmod(os.path.join(path,pixel+suffix+'.slurm'),0o755)

    return None


  
#create a FERRE control hash (content for a ferre input.nml file)
def mknml(synthfiles,root,k,order,path=None,nthreads=1):
    if path is None: path="."
    path=os.path.abspath(path)
    header=head_synth(synthfiles[0])
    nml={}
    ndim=int(header['N_OF_DIM'])
    nml['NDIM']=ndim
    nml['NOV']=ndim
    nml['INDV']=' '.join(map(str,arange(ndim)+1))
    for i in range(len(synthfiles)): nml['SYNTHFILE('+str(i+1)+')'] = "'"+os.path.join(path,synthfiles[i])+"'"
    nml['PFILE'] = "'"+root+".vrd"+"'"
    nml['FFILE'] = "'"+root+".frd"+"'"
    nml['ERFILE'] = "'"+root+".err"+"'"
    nml['OPFILE'] = "'"+root+".opf"+str(k)+"'"
    nml['OFFILE'] = "'"+root+".mdl"+str(k)+"'"
    nml['SFFILE'] = "'"+root+".nrd"+str(k)+"'"
    #nml['WFILE'] = "'"+root+".wav"+"'"
    nml['ERRBAR']=1
    nml['COVPRINT']=1
    #nml['WINTER']=2
    nml['INTER']=order
    nml['ALGOR']=3
    #nml['GEN_NUM']=5000
    #nml['NRUNS']=2**ndim
    #nml['INDINI']=''
    #for i in range(ndim): nml['INDINI']=nml['INDINI']+' 2 '
    nml['NTHREADS']=nthreads
    nml['F_FORMAT']=1
    nml['F_ACCESS']=0
    #nml['CONT']=1
    #nml['NCONT']=0
    nml['CONT']=3
    nml['NCONT']=100

    return nml

#write out a FERRE control hash to an input.nml file
def writenml(nml,nmlfile='input.nml',path=None):
    if path is None: path='./'
    f=open(os.path.join(path,nmlfile),'w')
    f.write('&LISTA\n')
    for item in nml.keys():
        f.write(str(item))
        f.write("=")
        f.write(str(nml[item]))
        f.write("\n")
    f.write(" /\n")
    f.close()
    return None

#run FERRE
def ferrerun(path=None):
    if path is None: path="./"
    pwd=os.path.abspath(os.curdir)
    os.chdir(path)
    ferre="/home/callende/ferre/src/a.out"
    code = subprocess.call(ferre)
    os.chdir(pwd)
    return code


#read redshift derived by the DESI pipeline
def readzbest(filename):
  hdu=fits.open(filename)
  if len(hdu) > 1:
    enames=extnames(hdu)
    print(enames)
    if 'ZBEST' in enames:
      zbest=hdu['zbest'].data
      targetid=zbest['targetid'] #array of long integers
    else:
      zbest=hdu[1].data
      plate=zbest['plate']
      mjd=zbest['mjd']
      fiberid=zbest['fiberid']
      targetid=[]
      for i in range(len(plate)): 
        targetid.append(str(plate[i])+'-'+str(mjd[i])+'-'+str(fiberid[i]))
      targetid=array(targetid)  #array of strings

    print(type(targetid),type(zbest['z']),type(targetid[0]))
    print(targetid.shape,zbest['z'].shape)
    z=dict(zip(targetid, zbest['z']))
  else:
    z=dict()

  return(z)

#read redshift derived by the Koposov pipeline
def readk(filename):
  clight=299792.458 #km/s
  hdu=fits.open(filename)
  if len(hdu) > 1:
    k=hdu[1].data
    targetid=k['targetid']
    #targetid=k['fiber']
    #teff=k['teff']
    #logg=k['loog']
    #vsini=k['vsini']
    #feh=k['feh']
    #z=k['vrad']/clight
    z=dict(zip(targetid, k['vrad']/clight))  
    #z_err=dict(zip(k['target_id'], k['vrad_err']/clight))  
  else:
    z=dict() 
  return(z)

#read truth tables (for simulations)
def readtruth(filename):
  hdu=fits.open(filename)
  truth=hdu[1].data
  targetid=truth['targetid']
  feh=dict(zip(targetid, truth['feh']))
  teff=dict(zip(targetid, truth['teff']))
  logg=dict(zip(targetid, truth['logg']))
  #rmag=dict(zip(targetid, truth['flux_r']))
  rmag=dict(zip(targetid, truth['mag']))
  z=dict(zip(targetid, truth['truez']))
  return(feh,teff,logg,rmag,z)

#read spectra
def readspec(filename,band=None):

  hdu=fits.open(filename)

  if filename.find('spectra-64') > -1 or filename.find('exp_') > -1: #DESI
    wavelength=hdu[band+'_WAVELENGTH'].data #wavelength array
    flux=hdu[band+'_FLUX'].data       #flux array (multiple spectra)
    ivar=hdu[band+'_IVAR'].data       #inverse variance (multiple spectra)
    #mask=hdu[band+'_MASK'].data       #mask (multiple spectra)
    res=hdu[band+'_RESOLUTION'].data  #resolution matrix (multiple spectra)
    #bintable=hdu['BINTABLE'].data  #bintable with info (incl. mag, ra_obs, dec_obs)

  if filename.find('spPlate') > -1: #SDSS/BOSS
    header=hdu['PRIMARY'].header
    wavelength=header['CRVAL1']+arange(header['NAXIS1'])*header['CD1_1'] 
#wavelength array
    wavelength=10.**wavelength
    flux=hdu['PRIMARY'].data       #flux array (multiple spectra)
    #ivar=hdu['IVAR'].data       #inverse variance (multiple spectra)    
    ivar=hdu[1].data       #inverse variance (multiple spectra)
    #andmask=hdu['ANDMASK'].data       #AND mask (multiple spectra)  
    #ormask=hdu['ORMASK'].data       #OR mask (multiple spectra)
    #res=hdu['WAVEDISP'].data  #FWHM array (multiple spectra)
    res=hdu[4].data  #FWHM array (multiple spectra)
    #bintable=hdu['BINTABLE'].data  #bintable with info (incl. mag, ra, dec)
    

  return((wavelength,flux,ivar,res))

#write piferre param. output
def write_tab_fits(pixel, path=None, pre='n'):
  
  if path is None: path=""
  root=os.path.join(path,pixel)
  o=glob.glob(root+".opf")
  m=glob.glob(root+".mdl")
  n=glob.glob(root+".nrd")
  fmp=glob.glob(root+".fmp.fits")
  
  success=[]
  fid=[]
  teff=[]
  logg=[]
  feh=[]
  alphafe=[]
  micro=[]
  covar=[]
  elem=[]
  elem_err=[]
  snr_med=[]
  chisq_tot=[]
  of=open(o[0],'r')
  for line in of:
    cells=line.split()
    #for N dim (since COVPRINT=1 in FERRE), there are m= 4 + N*(2+N) cells
    #and likewise we can calculate N = sqrt(m-3)-1
    m=len(cells)
    assert (m > 6), 'Error, the file '+o[0]+' has less than 7 columns, which would correspond to ndim=2'
    ndim=int(sqrt(m-3)-1)

    if (m == 19):
      #Kurucz grids with 3 dimensions: id, 3 par, 3 err, 0., 3x3 cov, med_snr, lchi
      #see Allende Prieto et al. (2018, A&A)
      feh.append(float(cells[1]))
      teff.append(float(cells[2]))
      logg.append(float(cells[3]))
      alphafe.append(nan)
      micro.append(nan)
      chisq_tot.append(10.**float(cells[9]))
      snr_med.append(float(cells[8]))
      if (pre == 'n'):
        cov = reshape(array(cells[10:],dtype=float),(3,3))
        covar.append(cov)
      else:
        print('Error: a 3 parameter grid was unexpectedly included among the *m* grids')
        sys.exit()

    elif (m == 39):
      #Kurucz grids with 5 dimensions: id, 5 par, 5 err, 0., 5x5 cov, med_snr, lchi
      #see Allende Prieto et al. (2018, A&A)
      feh.append(float(cells[1]))
      teff.append(float(cells[4]))
      logg.append(float(cells[5]))
      alphafe.append(float(cells[2]))
      micro.append(float(cells[3]))
      chisq_tot.append(10.**float(cells[13]))
      snr_med.append(float(cells[12]))
      if (pre == 'm'):
        cov = reshape(array(cells[14:],dtype=float),(5,5))
        covar.append(cov)
      else:
        print('Error: a 5 parameter grid was unexpectedly included among the *n* grids')
        sys.exit()

    elif (m == 12):
      #white dwarfs 2 dimensions: id, 2 par, 2err, 0.,2x2 cov, med_snr, lchi
      feh.append(-10.)
      teff.append(float(cells[1]))
      logg.append(float(cells[2]))
      alphafe.append(nan)
      micro.append(nan)
      chisq_tot.append(10.**float(cells[7]))
      snr_med.append(float(cells[6]))
      if (pre == 'n'):
        cov = zeros((3,3))
        cov[1:,1:] = reshape(array(cells[8:],dtype=float),(2,2))
        #cov = reshape(array(cells[8:],dtype=float),(2,2))
        covar.append(cov)    
      else:
        cov = zeros((5,5))
        cov[3:,3:] = reshape(array(cells[8:],dtype=float),(2,2))
        covar.append(cov)    
   


    if (chisq_tot[-1] < 1. and snr_med[-1] > 5.): # chi**2<10 and S/N>5
      success.append(1) 
    else: success.append(0)
    fid.append(cells[0])
    elem.append([nan,nan])
    elem_err.append([nan,nan])


  hdu0=fits.PrimaryHDU()
  now = datetime.datetime.fromtimestamp(time.time())
  nowstr = now.isoformat() 
  nowstr = nowstr[:nowstr.rfind('.')]
  hdu0.header['DATE'] = nowstr
  hdulist = [hdu0]

  #col01 = fits.Column(name='success',format='u1', array=array(success), unit='')
  #col02 = fits.Column(name='fid',format='30a',array=array(fid))  
  #col03 = fits.Column(name='teff',format='e4',array=array(teff))
  #col04 = fits.Column(name='logg',format='e4',array=array(logg))
  #col05 = fits.Column(name='feh',format='e4',array=array(feh))
  #col06 = fits.Column(name='alphafe',format='e4',array=array(alphafe))
  #col07 = fits.Column(name='micro',format='e4',array=array(micro))
  #col08 = fits.Column(name='covar',format='9e4',dim='(5, 5)',array=array(covar).reshape(len(success),5,5))
  #col09 = fits.Column(name='elem',format='2e4',dim='(2)',array=array(elem))
  #col10 = fits.Column(name='elem_err',format='2e4',dim='(2)',array=array(elem_err))
  #col11 = fits.Column(name='chisq_tot',format='e4',array=array(chisq_tot))
  #col12 = fits.Column(name='snr_med',format='e4',array=array(snr_med))

  #coldefs = fits.ColDefs([col01,col02,col03,col04,col05,col06,col07,col08,col09,col10,col11,col12])
  #hdu=fits.BinTableHDU.from_columns(coldefs)
  #hdu.header=header
  #hdulist.append(hdu)

  cols = {}
  cols['success'] = success
  cols['fid'] = fid
  cols['teff'] = array(teff)*units.K
  cols['logg'] = array(logg)
  cols['feh'] = array(feh)
  cols['alphafe'] = array(alphafe) 
  cols['micro'] = array(micro)*units.km/units.s
  if (pre == 'n'):
    cols['covar'] = array(covar).reshape(len(success),3,3)
  else:
    cols['covar'] = array(covar).reshape(len(success),5,5)
  cols['elem'] = array(elem)
  cols['elem_err'] = array(elem_err)
  cols['chisq_tot'] = array(chisq_tot)
  cols['snr_med'] = array(snr_med)

  colcomm = {
  'success': 'Bit indicating whether the code has likely produced useful results',
  'fid': 'Identifier used in FERRE to associate input/output files',
  'teff': 'Effective temperature',
  'logg': 'Surface gravity (g in cm/s**2)',
  'feh': 'Metallicity [Fe/H] = log10(N(Fe)/N(H)) - log10(N(Fe)/N(H))sun' ,
  'alphafe': 'Alpha-to-iron ratio [alpha/Fe]',
  'micro': 'Microturbulence',
  'covar': 'Covariance matrix for ([Fe/H], [a/Fe], logmicro, Teff,logg)',
  'elem': 'Elemental abundance ratios to iron [elem/Fe]',
  'elem_err': 'Uncertainties in the elemental abundance ratios to iron',
  'chisq_tot': 'Total chi**2',
  'snr_med': 'Median signal-to-ratio'
  }      

  
  table = tbl.Table(cols)
  hdu=fits.BinTableHDU(table,name = 'SPTAB')
  #hdu.header['EXTNAME']= ('SPTAB', 'Stellar Parameter Table')
  i = 0
  for entry in colcomm.keys():
    print(entry) 
    hdu.header['TCOMM'+str(i+1)] = colcomm[entry]
    i+=1
  hdulist.append(hdu)


  if len(fmp) > 0:
    ff=fits.open(fmp[0])
    fibermap=ff[1]
    hdu=fits.BinTableHDU.from_columns(fibermap, name='FIBERMAP')
    #hdu.header['EXTNAME']='FIBERMAP'
    hdulist.append(hdu)

  hdul=fits.HDUList(hdulist)
  hdul.writeto('sptab-64-'+root+'.fits')
  
  return None
  
#write piferre spec. output  
def write_mod_fits(pixel, path=None):  
  
  if path is None: path=""
  root=os.path.join(path,pixel)
  
  xbandfiles = sorted(glob.glob(root+'-*.wav'))
  band = []
  npix = []
  for entry in xbandfiles:
    match = re.search('-[\w]*.wav',entry)
    tag = match.group()[1:-4]
    if match: band.append(tag.upper())
    x = loadtxt(root+'-'+tag+'.wav')
    npix.append(len(x))
    
  x = loadtxt(root+'.wav')
  if len(npix) == 0: npix.append(len(x))

  m=glob.glob(root+".mdl")
  e=glob.glob(root+".err")
  n=glob.glob(root+".nrd")

  fmp=glob.glob(root+".fmp.fits")  
  mdata=loadtxt(m[0])
  edata=loadtxt(e[0])
  if (len(n) > 0): 
    odata=loadtxt(n[0])
    f=glob.glob(root+".frd")
    fdata=loadtxt(f[0])
    edata=edata/fdata*odata
  else:
    odata=loadtxt(root+".frd")  

  hdu0=fits.PrimaryHDU()
  now = datetime.datetime.fromtimestamp(time.time())
  nowstr = now.isoformat() 
  nowstr = nowstr[:nowstr.rfind('.')]
  hdu0.header['DATE'] = nowstr
  hdulist = [hdu0]

  i = 0
  j1 = 0

  for entry in band:
    j2 = j1 + npix[i] 
    print(entry,i,npix[i],j1,j2)
    #colx = fits.Column(name='wavelength',format='e8', array=array(x[j1:j2]))
    #coldefs = fits.ColDefs([colx])
    #hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu = fits.ImageHDU(name=entry+'_WAVELENGTH', data=x[j1:j2])
    #hdu.header['EXTNAME']=entry+'_WAVELENGTH'
    hdulist.append(hdu)
    
    if odata.ndim == 2: tdata = odata[:,j1:j2]
    else: tdata = odata[j1:j2][None,:]
    col01 = fits.Column(name='obs',format=str(npix[i])+'e8', dim='('+str(npix[i])+')', array=tdata)
    if edata.ndim == 2: tdata = edata[:,j1:j2]
    else: tdata = edata[j1:j2][None,:]
    col02 = fits.Column(name='err',format=str(npix[i])+'e8', dim='('+str(npix[i])+')', array=tdata)
    if mdata.ndim == 2: tdata = mdata[:,j1:j2]
    else: tdata = mdata[j1:j2][None,:]
    col03 = fits.Column(name='fit',format=str(npix[i])+'e8', dim='('+str(npix[i])+')', array=tdata)    
    coldefs = fits.ColDefs([col01,col02,col03])
    hdu=fits.BinTableHDU.from_columns(coldefs, name=entry+'_MODEL')
    #hdu = fits.ImageHDU(name=entry+'_MODEL', data=stack([odata[:,j1:j2],edata[:,j1:j2],mdata[:,j1:j2]]) ) 
    #hdu.header['EXTNAME']=entry+'_MODEL'
    hdulist.append(hdu)
    i += 1
    j1 = j2

  if len(fmp) > 0:
    ff=fits.open(fmp[0])
    fibermap=ff[1]
    hdu=fits.BinTableHDU.from_columns(fibermap, name='FIBERMAP')
    #hdu.header['EXTNAME']='FIBERMAP'
    hdulist.append(hdu)

  hdul=fits.HDUList(hdulist)
  hdul.writeto('spmod-64-'+root+'.fits')
  
  return None

#write ferre files
def write_ferre_input(root,ids,par,y,ey,path=None,suffix=''):

  if path is None: path="./"

  #open ferre input files
  vrd=open(os.path.join(path,root)+suffix+'.vrd','w')
  frd=open(os.path.join(path,root)+suffix+'.frd','w')
  err=open(os.path.join(path,root)+suffix+'.err','w')

  nspec, freq = y.shape

  #loop to write data files
  i=0
  while (i < nspec):

    print(str(i)+' ')

    #vrd.write("target_"+str(i+1)+" 0.0 0.0 0.0")
    ppar=[ids[i]]
    for item in par[ids[i]]: ppar.append(item)
    #vrd.write(' '.join(map(str,ppar))
    vrd.write("%30s %6.2f %10.2f %6.2f %6.2f %12.9f %12.9f %12.9f %12.9f\n" % 
tuple(ppar) )    
    #ppar.tofile(ppar,sep=" ",format="%s")
    #vrd.write("\n")

    yy=y[i,:]
    yy.tofile(frd,sep=" ",format="%0.4e")
    frd.write("\n")
    eyy=ey[i,:]
    eyy.tofile(err,sep=" ",format="%0.4e")
    err.write("\n")
    i+=1
    #print(i,yy[0],eyy[0])
 
  #close files
  vrd.close()
  frd.close()
  err.close()


def opfmerge(pixel,path=None,wait_on_sorted=False,pre='n'):

  if path is None: path="./"
  root=os.path.join(path,pixel)

  if wait_on_sorted:
    o=sorted(glob.glob(root+".opf*_sorted"))  
    while (len(o) > 0):
      time.sleep(5)
      o=sorted(glob.glob(root+".opf*_sorted"))
      

  o=sorted(glob.glob(root+".opf?"))
  m=sorted(glob.glob(root+".mdl?"))
  n=sorted(glob.glob(root+".nrd?"))
  
  llimit = [3500.,5500.,7000.,10000.,20000.,6000.,10000.,10000.,15000.]
  if (pre == 'n'):
    iteff = [2,     2,     2,    2,     2,      1,    1,     1,     1   ]
    ilchi = [9,     9,     9,    9,     9,      7,    7,     7,     7   ]
  elif (pre == 'm'):
    iteff = [4,     4,     4,    4,     4,      1,    1,     1,     1   ]
    ilchi = [13,    13,    13,   13,    13,     7,    7,     7,     7   ]
  else:
    print('Error: pre is neither n or m -- unknown grid family')
    sys.exit()

  ngrid=len(o)
  if ngrid != len(m): 
    print("there are different number of opf? and mdl? arrays")
    return(0)
  if (len(n) > 0):
    if ngrid != len(m):  
      print("there are different number of opf? and mdl? arrays")
      return(0)

  #open input files
  of=[]
  mf=[]
  if len(n) > 0: nf=[]
  for i in range(len(o)):
    of.append(open(o[i],'r'))
    mf.append(open(m[i],'r'))
    if len(n) > 0: nf.append(open(n[i],'r'))
  print(o)
  print(of)
  #open output files
  oo=open(root+'.opf','w')
  mo=open(root+'.mdl','w')
  if len(n) > 0: no=open(root+'.nrd','w')
 
  for line in of[0]: 
    array=line.split()
    min_chi=float(array[ilchi[0]])
    min_oline=line
    print(min_chi,min_oline)
    min_mline=mf[0].readline()
    if len(n) > 0: min_nline=nf[0].readline()
    for i in range(len(o)-1):
      oline=of[i+1].readline()
      mline=mf[i+1].readline()
      if len(n) > 0: nline=nf[i+1].readline()
      array=oline.split()
      #print(len(array))
      #print(array)
      #print(i,ilchi[i+1],len(array))
      #print(i,float(array[ilchi[i+1]]))
      if float(array[ilchi[i+1]]) < min_chi and float(array[iteff[i+1]]) > llimit[i+1]*1.01: 
        min_chi=float(array[ilchi[i+1]])
        min_oline=oline
        min_mline=mline
        if len(n) > 0: min_nline=nline
    
    #print(min_chi,min_oline)
    oo.write(min_oline)
    mo.write(min_mline)
    if len(n) > 0: no.write(min_nline)
  
  #close input files
  for i in range(len(o)):
    #print(o[i],m[i])
    of[i].close
    mf[i].close
    if len(n) > 0: nf[i].close

  #close output files
  oo.close
  mo.close
  if len(n) > 0: no.close
  
  return None

#get names of extensions from a FITS file
def extnames(hdu):
  #hdu must have been open as follows hdu=fits.open(filename)
  x=hdu.info(output=False)
  names=[]
  for entry in x: names.append(entry[1])
  return(names)

#identify input data files and associated zbest files 
def finddatafiles(path,pixel,sdir='',rvpath=None):

  if rvpath is None: rvpath = path

  infiles=os.listdir(os.path.join(path,sdir,pixel))  
  datafiles=[]
  zbestfiles=[]
  for ff in infiles: #add subdirs, which may contain zbest files for SDSS/BOSS
    if os.path.isdir(os.path.join(path,sdir,pixel,ff)): 
      extrafiles=os.listdir(os.path.join(path,sdir,pixel,ff))
      for ff2 in extrafiles: 
        infiles.append(os.path.join(ff,ff2))

  if rvpath != path:
    if os.path.isdir(os.path.join(rvpath,sdir,pixel)):
      infiles2=os.listdir(os.path.join(rvpath,sdir,pixel))
      for ff in infiles2: #add subdirs, which may contain zbest files for SDSS/BOSS
        infiles.append(ff)
        if os.path.isdir(os.path.join(rvpath,sdir,pixel,ff)): 
          extrafiles2=os.listdir(os.path.join(rvpath,sdir,pixel,ff))
          for ff2 in extrafiles2: 
            infiles.append(os.path.join(ff,ff2))


  infiles.sort()
  print('infiles=',infiles)

  for filename in infiles:
# DESI sims/data
    if (filename.find('spectra-64') > -1 and filename.find('.fits') > -1):
      datafiles.append(os.path.join(path,sdir,pixel,filename))
    elif (filename.find('zbest-64') > -1 and filename.find('.fits') > -1):
      zbestfiles.append(os.path.join(rvpath,sdir,pixel,filename))
# BOSS data
    elif (filename.find('spPlate') > -1 and filename.find('.fits') > -1):
      datafiles.append(os.path.join(path,sdir,pixel,filename))
    elif (filename.find('spZbest') > -1 and filename.find('.fits') > -1):
      zbestfiles.append(os.path.join(rvpath,sdir,pixel,filename))
#  DESI commissioning data
    elif (filename.find('rvtab') > -1 and filename.find('.fits') > -1):
      zbestfiles.append(os.path.join(rvpath,sdir,pixel,filename))


  #analyze the situation wrt input files
  ndatafiles=len(datafiles)
  nzbestfiles=len(zbestfiles)
  print ('Found '+str(ndatafiles)+' input data files')
  for filename in datafiles: print(filename+'--')
  print ('and '+str(nzbestfiles)+' associated zbest files')
  for filename in zbestfiles: print(filename+'--')

  if (ndatafiles != nzbestfiles):
    print('ERROR -- there is a mismatch between the number of data files and zbest files, this pixel is skipped')
    return (None,None)

  return (datafiles,zbestfiles)


#pack a collection of fits files with binary tables in multiple HDUs into a single one
def packfits(input="*.fits",output="output.fits"):


  f = glob.glob(input)

  print('reading ... ',f[0])
  hdul1 = fits.open(f[0])
  hdu0 = hdul1[0]
  for entry in f[1:]:       
    print('reading ... ',entry)
    hdul2 = fits.open(entry)
    for i in arange(len(hdul1)-1)+1:
      nrows1 = hdul1[i].data.shape[0]
      nrows2 = hdul2[i].data.shape[0]
      nrows = nrows1 + nrows2
      hdu = fits.BinTableHDU.from_columns(hdul1[i].columns, nrows=nrows)
      hdu.header['EXTNAME'] = hdul1[i].header['EXTNAME']
      for colname in hdul1[i].columns.names:
        hdu.data[colname][nrows1:] = hdul2[i].data[colname] 

      if i == 1: 
        hdu1 = hdu 
      else: 
        hdu2 = hdu 

    hdul1 = fits.HDUList([hdu0,hdu1,hdu2])

  hdul1.writeto(output)

  return(None)


#process a single pixel
def do(path,pixel,sdir='',truth=None,nthreads=1,rvpath=None, pre='n'):
  
  #get input data files
  datafiles,zbestfiles  = finddatafiles(path,pixel,sdir,rvpath=rvpath) 
  if (datafiles == None or zbestfiles == None): return None

  #loop over possible multiple data files in the same pixel
  for i in range(len(datafiles)):

    datafile=datafiles[i]
    zbestfile=zbestfiles[i]

    #get redshifts
    if zbestfile.find('best') > -1:
      z=readzbest(zbestfile)
    else:
      #Koposov pipeline
      z=readk(zbestfile)
  
    #read primary header and  
    #find out if there is FIBERMAP extension
    #identify MWS targets
    hdu=fits.open(datafile)
    enames=extnames(hdu)
    pheader=hdu['PRIMARY'].header
    print('datafile='+datafile)
    print('extensions=',enames)

    if 'FIBERMAP' in enames: #DESI data
      fibermap=hdu['FIBERMAP']
      suffix=''
      targetid=fibermap.data['TARGETID']
      if 'RA_TARGET' in fibermap.data.names: 
        ra=fibermap.data['RA_TARGET']
      else:
        if 'TARGET_RA' in fibermap.data.names:
          ra=fibermap.data['TARGET_RA']
        else:
          ra=-9999.*ones(len(targetid))
      if 'DEC_TARGET' in fibermap.data.names:
        dec=fibermap.data['DEC_TARGET']
      else:
        if 'TARGET_DEC' in fibermap.data.names:
          dec=fibermap.data['TARGET_DEC']
        else:
          dec=-9999.*ones(len(targetid))
      if 'MAG' in fibermap.data.names: 
        mag=fibermap.data['MAG']
      else:
          mag=[-9999.*ones(5)]
          for kk in range(len(targetid)-1): mag.append(-9999.*ones(5))
      nspec=ra.size
      
      #set the set of grids to be used
      grids=[]
      for item in range(9): grids.append(pre+'_rdesi'+str(item+1))    
      #for item in range(10): grids.append('n_rdesi'+str(item+1))
      print('grids=',grids)
      if (pre == 'n'):
        maxorder=[3,3,3,2,1,3,3,3,3] #max. order that can be used for interpolations
      elif (pre == 'm'):
        maxorder=[3,3,3,3,2,3,3,3,3]
      else:
        print('Error: pre is neither n or m -- unknown grid family')
        sys.exit()
      #bands=['b']
      bands=['b','r','z']

    else:  #SDSS/BOSS data

      plate=pheader['PLATEID']
      mjd=pheader['MJD']
      suffix="-"+str(mjd)
      #fibermap=hdu['PLUGMAP']
      fibermap=hdu[5]
      fiberid=fibermap.data['fiberid']
      ra=fibermap.data['RA']
      dec=fibermap.data['DEC']
      #mag=zeros((ra.size,5)) # used zeros for LAMOST fibermap.data['MAG']
      mag=fibermap.data['MAG']
      nspec=ra.size
      targetid=[]
      for i in range(nspec): 
        targetid.append(str(plate)+'-'+str(mjd)+'-'+str(fiberid[i]))

      targetid=array(targetid)


      #set the set of grids to be used
      #SDSS/BOSS
      grids=[]
      for item in range(9): grids.append(pre+'_rboss'+str(item+1))
      print(grids)
      if (pre == 'n'):
        maxorder=[3,3,3,2,1,3,3,3,3] #max. order that can be used for interpolations
      elif (pre == 'm'):
        maxorder=[3,3,3,3,2,3,3,3,3]
      else:
        print('Error: pre is neither n or m -- unknown grid family')
        sys.exit()

      bands=['']
      
      #LAMOST
      #grids=['n_crump3hL']
      #maxorder=[3]
      #bands=['']


    #identify targets to process based on redshift: 0.00<=|z|<0.01
    process_target = zeros(nspec, dtype=bool)
    for i in range(nspec):
      if z.get(targetid[i],-1) != -1:
        if (abs(z[targetid[i]]) < 0.01) & (abs(z[targetid[i]]) >= 0.): process_target[i]= True

    
    #skip the rest of the code if there are no targets
    if (process_target.nonzero()[0].size == 0): return None


    #set ids array (with targetids) and par dictionary for vrd/ipf file
    ids=[]
    par={}

    #truth (optional, for simulations)
    #if (len(sys.argv) == 3):
    npass=0 #count targets that pass the filter (process_target)
    if truth is not None:
      (true_feh,true_teff,true_logg,true_rmag,true_z)=truth
      for k in range(nspec):
        if process_target[k]:
          npass=npass+1
          id=str(targetid[k])
          ids.append(id)
          par[id]=[true_feh[targetid[k]],true_teff[targetid[k]],
                         true_logg[targetid[k]],true_rmag[targetid[k]],
			 true_z[targetid[k]],z[targetid[k]],
                         ra[k],dec[k]]
          #stop
    else:
      for k in range(nspec):
        if process_target[k]:
          npass=npass+1
          id=str(targetid[k])
          ids.append(id)
          #we patch the redshift here to handle missing redshifts for comm. data from Sergey
          #z[targetid[k]]=0.0
          par[id]=[0.0,0.0,0.0,mag[k][2],0.0,z[targetid[k]],ra[k],dec[k]]
          #stop        

    #collect data for each band
    for j in range(len(bands)):

      if bands[j] == '': 
        gridfile=grids[0]+'.dat'
      else:
        gridfile=grids[0]+'-'+bands[j]+'.dat'

      #read grid wavelength array
      x1=lambda_synth(gridfile)

      #read DESI data, select targets, and resample 
      (x,y,ivar,r)=readspec(datafile,bands[j])
      ey=sqrt(divide(1.,ivar,where=(ivar != 0.)))
      ey[where(ivar == 0.)]=max(y)*1e3

      #plt.ion()
      #plt.plot(x,y[0])
      #plt.show()
      #plt.plot(x,y[0])
      #plt.show()

      nspec, freq = y.shape
      print('nspec=',nspec)    
      print('n(process_target)=',process_target.nonzero()[0].size)
      y2=zeros((npass,len(x1)))
      ey2=zeros((npass,len(x1)))
      k2=0
      print('nspec,len(z),npass,len(x1)=',nspec,len(z),npass,len(x1))
      for k in range(nspec):
        if process_target[k]:
          y2[k2,:]=interp(x1,x*(1.-z[targetid[k]]),y[k,:])
          ey2[k2,:]=interp(x1,x*(1-z[targetid[k]]),ey[k,:])
          k2=k2+1

      if (j==0):
        xx=x1
        yy=y2
        eyy=ey2
      else:
        xx=concatenate((xx,x1))
        yy=concatenate((yy,y2),axis=1)
        eyy=concatenate((eyy,ey2),axis=1)

      savetxt(os.path.join(sdir,pixel,pixel)+suffix+'-'+bands[j]+'.wav',x1,fmt='%14.5e')

    savetxt(os.path.join(sdir,pixel,pixel)+suffix+'.wav',xx,fmt='%14.5e')
    fmp = tbl.Table(fibermap.data) [process_target]
    hdu0 = fits.BinTableHDU(fmp)
    hdu0.writeto(os.path.join(sdir,pixel,pixel)+suffix+'.fmp.fits')

    write_ferre_input(pixel,ids,par,yy,eyy,path=os.path.join(sdir,pixel),suffix=suffix)

    #write slurm script
    write_slurm(pixel,path=os.path.join(sdir,pixel),
            ngrids=len(grids),nthreads=nthreads, suffix=suffix, pre=pre)


    #loop over all grids
    for k in range(len(grids)):
 
      #make an array with the names of the synthfiles
      synthfiles=[]
      for j in range(len(bands)):
        if bands[j] == '': 
          gridfile=grids[k]+'.dat'
        else:
          gridfile=grids[k]+'-'+bands[j]+'.dat'

        synthfiles.append(gridfile)

      #prepare ferre control file
      nml=mknml(synthfiles,pixel+suffix,k,maxorder[k],nthreads=nthreads)
      writenml(nml,nmlfile='input.nml'+suffix+'_'+str(k),path=os.path.join(sdir,pixel))
      writenml(nml,path=os.path.join(sdir,pixel))

      print(k)
      print(nml)

      #run ferre
      #ferrerun(path=os.path.join(sdir,pixel))

    #opfmerge(pixel,path=os.path.join(sdir,pixel))


  return None

#find pixels in 'root' directory (spectra-64)
def getpixels(root):
  #root='spectro/redux/dc17a2/spectra-64/'
  d1=os.listdir(root)
  d=[]
  for x in d1:
    d2=os.listdir(os.path.join(root,x))
    #d.append(os.path.join(root,x))
    res=[i for i in d2 if '.fits' in i] 
    for y in d2: 
      #d.append(os.path.join(root,x))
      if len(res) == 0: # there are no fits files in the 1st directory, so 2 layer (e.g. DESI)
        d.append(os.path.join(root,x,y))
      else: 
        entry=os.path.join(root,x)
        if entry not in d: d.append(entry)  #keep only the first layer (SDSS/BOSS)

  print(d)
  print(len(d))
  return(d)

#run
def run(pixel,path=None):
  if path is None: path="./"
  #directly run
  #pwd=os.path.abspath(os.curdir)
  #os.chdir(path)
  #job="/bin/bash "+pixel+'.slurm'
  #code=subprocess.call(job)
  #os.chdir(pwd)
  #slurm
  job="sbatch "+os.path.join(path,pixel+'.slurm')
  code=subprocess.call(job)
  #kiko
  #job="kiko "+os.path.join(path,pixel+'.slurm')
  #code=subprocess.call(job)

  return code



if __name__ == "__main__":

  nthreads=1
  pre='n'

  path=sys.argv[1]
  rvpath=path
  #path is the path to the spectra-64 directory
  pixels=getpixels(path)
  print(pixels)
  

  if (len(sys.argv) == 3):  
    truthfile=sys.argv[2] 
    truthtuple=readtruth(truthfile)
  else: truthtuple=None

  for entry in pixels:
    head, pixel = os.path.split(entry)
    print('head/pixel=',head,pixel)
    sdir=''
    if head.find('spectra-64') > -1: 
      sdir=pixel[:-2]
    if head.find('spectra_fake_64') > -1:
      head, sdir = os.path.split(head)
      rvpath = os.path.join(path,"../rv_output")
      if not os.path.exists(sdir): os.mkdir(sdir)
    if sdir != '': 
      if not os.path.exists(sdir):os.mkdir(sdir)
    if not os.path.exists(os.path.join(sdir,pixel)):os.mkdir(os.path.join(sdir,pixel))
    do(path,pixel,sdir=sdir,truth=truthtuple,nthreads=nthreads, rvpath=rvpath, pre=pre)
    #run(pixel,path=os.path.join(sdir,pixel))


