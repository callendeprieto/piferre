#!/home/callende/anaconda3/bin/python3
'''
Interface to use FERRE from python for DESI/BOSS/WEAVE data

use: piferre -p path-to-spectra [-t truthfile -rv rvpath -m models -l libpath -n nthreads]

e.g. piferre -p /data/spectro/redux/dc17a2/spectra-64 

Author C. Allende Prieto
'''
import pdb
import sys
import os
import glob
import re
from numpy import arange,loadtxt,savetxt,zeros,ones,nan,sqrt,interp,concatenate,array,reshape,min,max,where,divide,mean, stack, vstack
from astropy.io import fits
import astropy.table as tbl
import astropy.units as units
import matplotlib.pyplot as plt
import subprocess
import datetime, time
import argparse
import yaml

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
def write_slurm(root,nthreads=1,path=None,ngrids=None, config='desi-n.yaml'):
    ferre=os.environ['HOME']+"/ferre/src/a.out"
    python_path=os.environ['HOME']+"/piferre"
    try: 
      host=os.environ['HOST']
    except:
      host='Unknown'

    now=time.strftime("%c")
    if path is None: path='.'
    if ngrids is None: ngrids=1

    f=open(os.path.join(path,root+'.slurm'),'w')
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
      f.write("#SBATCH  -J "+str(root)+" \n")
      f.write("#SBATCH  -p batch"+" \n")
      f.write("#SBATCH  -o "+str(root)+"_%j.out"+" \n")
      f.write("#SBATCH  -e "+str(root)+"_%j.err"+" \n")
      f.write("#SBATCH  -n "+str(nthreads)+" \n")
      f.write("#SBATCH  -t 04:00:00"+" \n") #hh:mm:ss
      f.write("#SBATCH  -D "+os.path.abspath(path)+" \n")
    f.write("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# \n")
    f.write("export OMP_NUM_THREADS="+str(nthreads)+"\n")
    f.write("cd "+os.path.abspath(path)+"\n")
    for i in range(ngrids):
      #f.write("cp input.nml-"+root+"_"+str(i)+" input.nml \n")
      f.write("time "+ferre+" -l input.lst-"+root+"_"+str(i)+" >& log_"+str(i))
      #if (i == 8): 
      #  f.write( "  \n")
      #else:
      #  f.write( " & \n")
      f.write( " & \n")
    if ngrids > 1:
      f.write("wait \n")
      f.write("python3 -c \"import sys; sys.path.insert(0, '"+python_path+ \
              "'); from piferre import opfmerge, write_tab_fits, write_mod_fits; opfmerge(\'"+\
              str(root)+"\',config='"+config+"\'); write_tab_fits(\'"+\
              str(root)+"\',config='"+config+"\'); write_mod_fits(\'"+\
              str(root)+"\')\"\n")
    f.close()
    os.chmod(os.path.join(path,root+'.slurm'),0o755)

    return None


  
#create a FERRE control hash (content for a ferre input.nml file)
def mknml(conf,root,nthreads=1,libpath='.',path='.'):

  grids=conf['grids']

  for k in range(len(grids)): #loop over all grids
    synth=grids[k]
    synthfiles=[]
    for band in conf['bands']:
      if band == '':
        gridfile=synth+'.dat'
      else:
        gridfile=synth+'-'+band+'.dat'
      synthfiles.append(gridfile)

    libpath=os.path.abspath(libpath)
    header=head_synth(os.path.join(libpath,synthfiles[0]))
    ndim=int(header['N_OF_DIM'])

    lst=open(os.path.join(path,'input.lst-'+root+'_'+str(k)),'w')
    for run in conf[synth]: #loop over all runs (param + elements)
      print('run=',run)
      nml={}
      nml['NDIM']=ndim
      nml['NOV']=conf[synth][run]['nov']
      #nml['INDV']=' '.join(map(str,arange(ndim)+1))
      nml['INDV']=' '.join(map(str,conf[synth][run]['indv']))
      for i in range(len(synthfiles)): 
        nml['SYNTHFILE('+str(i+1)+')'] = "'"+os.path.join(libpath,synthfiles[i])+"'"
      nml['PFILE'] = "'"+root+'.'+conf[synth][run]['pfile_ext']+"'"
      nml['FFILE'] = "'"+root+'.'+conf['global']['ffile_ext']+"'"
      nml['ERFILE'] = "'"+root+'.'+conf['global']['erfile_ext']+"'"
      nml['OPFILE'] = "'"+root+'.'+conf[synth][run]['opfile_ext']+"'"
      nml['OFFILE'] = "'"+root+'.'+conf[synth][run]['offile_ext']+"'"
      nml['SFFILE'] = "'"+root+'.'+conf[synth][run]['sffile_ext']+"'"
      #nml['WFILE'] = "'"+root+".wav"+"'"
      nml['ERRBAR']=conf['global']['errbar']
      nml['COVPRINT']=conf['global']['covprint']
      #nml['WINTER']=2
      nml['INTER']=conf['global']['inter']
      nml['ALGOR']=conf['global']['algor']
      #nml['GEN_NUM']=5000
      #nml['NRUNS']=2**ndim
      #nml['INDINI']=''
      #for i in range(ndim): nml['INDINI']=nml['INDINI']+' 2 '
      nml['NTHREADS']=nthreads
      nml['F_FORMAT']=conf['global']['f_format']
      nml['F_ACCESS']=conf['global']['f_access']
      #nml['CONT']=1
      #nml['NCONT']=0
      nml['CONT']=conf[synth][run]['cont']
      nml['NCONT']=conf[synth][run]['ncont']

      nmlfile='input.nml-'+root+'_'+str(k)+run
      lst.write(nmlfile+'\n')
      write_nml(nml,nmlfile=nmlfile,path=path)

    lst.close()

  return None

#write out a FERRE control hash to an input.nml file
def write_nml(nml,nmlfile='input.nml',path=None):
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

  if filename.find('spectra-') > -1 or filename.find('exp_') > -1: #DESI
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
def write_tab_fits(root, path=None, config='desi-n.yaml'):
  
  if path is None: path=""
  proot=os.path.join(path,root)
  o=glob.glob(proot+".opf")
  m=glob.glob(proot+".mdl")
  n=glob.glob(proot+".nrd")
  fmp=glob.glob(proot+".fmp.fits")
  
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

    if (ndim == 3):
      #Kurucz grids with 3 dimensions: id, 3 par, 3 err, 0., med_snr, lchi, 3x3 cov
      #see Allende Prieto et al. (2018, A&A)
      feh.append(float(cells[1]))
      teff.append(float(cells[2]))
      logg.append(float(cells[3]))
      alphafe.append(nan)
      micro.append(nan)
      chisq_tot.append(10.**float(cells[9]))
      snr_med.append(float(cells[8]))
      cov = reshape(array(cells[10:],dtype=float),(3,3))
      covar.append(cov)

    elif (ndim == 5):
      #Kurucz grids with 5 dimensions: id, 5 par, 5 err, 0., med_snr, lchi, 5x5 cov
      #see Allende Prieto et al. (2018, A&A)
      feh.append(float(cells[1]))
      teff.append(float(cells[4]))
      logg.append(float(cells[5]))
      alphafe.append(float(cells[2]))
      micro.append(float(cells[3]))
      chisq_tot.append(10.**float(cells[13]))
      snr_med.append(float(cells[12]))
      cov = reshape(array(cells[14:],dtype=float),(5,5))
      covar.append(cov)
  
    elif (ndim == 2):
      #white dwarfs 2 dimensions: id, 2 par, 2err, 0., med_snr, lchi, 2x2 cov
      feh.append(-10.)
      teff.append(float(cells[1]))
      logg.append(float(cells[2]))
      alphafe.append(nan)
      micro.append(nan)
      chisq_tot.append(10.**float(cells[7]))
      snr_med.append(float(cells[6]))
      if (config == 'desi-n.yaml'):
        cov = zeros((3,3))
        cov[1:,1:] = reshape(array(cells[8:],dtype=float),(2,2))
        #cov = reshape(array(cells[8:],dtype=float),(2,2))
        covar.append(cov)    
      else:
        cov = zeros((5,5))
        cov[3:,3:] = reshape(array(cells[8:],dtype=float),(2,2))
        covar.append(cov)    
   
    elif (ndim == 4):
      #Phoenix grid from Sergey, with 4 dimensions: id, 4 par, 4err, 0., med_snr, lchi, 4x4 cov
      feh.append(float(cells[2]))
      teff.append(float(cells[4]))
      logg.append(float(cells[3]))
      alphafe.append(float(cells[1]))
      micro.append(nan)
      chisq_tot.append(10.**float(cells[11]))
      snr_med.append(float(cells[10]))
      if (config == 'desi-n.yaml'):
        cov = zeros((3,3))
        cov[:,:] = reshape(array(cells[12:],dtype=float),(4,4))[1:,1:]
        covar.append(cov)    
      else:
        print('Error: this path in the code is not yet ready!')
        sys.exit()
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
  cols['SUCCESS'] = success
  cols['FID'] = fid
  cols['TEFF'] = array(teff)*units.K
  cols['LOGG'] = array(logg)
  cols['FEH'] = array(feh)
  cols['ALPHAFE'] = array(alphafe) 
  cols['MICRO'] = array(micro)*units.km/units.s
  if (config == 'desi-n.yaml'):
    cols['COVAR'] = array(covar).reshape(len(success),3,3)
  else:
    cols['COVAR'] = array(covar).reshape(len(success),5,5)
  cols['ELEM'] = array(elem)
  cols['ELEM_ERR'] = array(elem_err)
  cols['CHISQ_TOT'] = array(chisq_tot)
  cols['SNR_MED'] = array(snr_med)

  colcomm = {
  'success': 'Bit indicating whether the code has likely produced useful results',
  'FID': 'Identifier used in FERRE to associate input/output files',
  'TEFF': 'Effective temperature',
  'LOGG': 'Surface gravity (g in cm/s**2)',
  'FEH': 'Metallicity [Fe/H] = log10(N(Fe)/N(H)) - log10(N(Fe)/N(H))sun' ,
  'ALPHAFE': 'Alpha-to-iron ratio [alpha/Fe]',
  'MICRO': 'Microturbulence',
  'COVAR': 'Covariance matrix for ([Fe/H], [a/Fe], logmicro, Teff,logg)',
  'ELEM': 'Elemental abundance ratios to iron [elem/Fe]',
  'ELEM_ERR': 'Uncertainties in the elemental abundance ratios to iron',
  'CHISQ_TOT': 'Total chi**2',
  'SNR_MED': 'Median signal-to-ratio'
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
  hdul.writeto('sptab_'+root+'.fits')
  
  return None
  
#write piferre spec. output  
def write_mod_fits(root, path=None):  
  
  if path is None: path=""
  proot=os.path.join(path,root)
  
  xbandfiles = sorted(glob.glob(proot+'-*.wav'))
  band = []
  npix = []
  for entry in xbandfiles:
    match = re.search('-[\w]*.wav',entry)
    tag = match.group()[1:-4]
    if match: band.append(tag.upper())
    x = loadtxt(proot+'-'+tag+'.wav')
    npix.append(len(x))
    
  x = loadtxt(proot+'.wav')
  if len(npix) == 0: npix.append(len(x))

  m=glob.glob(proot+".mdl")
  e=glob.glob(proot+".err")
  n=glob.glob(proot+".nrd")

  fmp=glob.glob(proot+".fmp.fits")  
  mdata=loadtxt(m[0])
  edata=loadtxt(e[0])
  if (len(n) > 0): 
    odata=loadtxt(n[0])
    f=glob.glob(proot+".frd")
    fdata=loadtxt(f[0])
    edata=edata/fdata*odata
  else:
    odata=loadtxt(proot+".frd")  

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
  hdul.writeto('spmod_'+root+'.fits')
  
  return None

#write ferre files
def write_ferre_input(root,ids,par,y,ey,path=None):

  if path is None: path="./"

  #open ferre input files
  vrd=open(os.path.join(path,root)+'.vrd','w')
  frd=open(os.path.join(path,root)+'.frd','w')
  err=open(os.path.join(path,root)+'.err','w')

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


def opfmerge(root,path=None,wait_on_sorted=False,config='desi-n.yaml'):

  if path is None: path="./"
  proot=os.path.join(path,root)

  if wait_on_sorted:
    o=sorted(glob.glob(proot+".opf*_sorted"))  
    while (len(o) > 0):
      time.sleep(5)
      o=sorted(glob.glob(proot+".opf*_sorted"))
      

  o=sorted(glob.glob(proot+".opf?"))
  m=sorted(glob.glob(proot+".mdl?"))
  n=sorted(glob.glob(proot+".nrd?"))
 

  llimit=[] # lower limits for Teff
  iteff=[]  # column for Teff in opf
  ilchi=[]  # column for log10(red. chi**2) in opf
  ydir = os.path.dirname(os.path.realpath(__file__))
  yfile=open(os.path.join(ydir,config),'r')
  conf=yaml.full_load(yfile)
  yfile.close()
  #set the set of grids to be used
  grids=conf['grids']
  for entry in grids:
    tmplist=conf[entry]['param']['labels']
    iteffcol = [idx for idx, element in enumerate(tmplist) if element == 'Teff'][0]
    llimit.append(conf[entry]['param']['llimits'][iteffcol])
    iteff.append(iteffcol+1)
    ilchi.append(conf[entry]['param']['ndim']*2+3)


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
  oo=open(proot+'.opf','w')
  mo=open(proot+'.mdl','w')
  if len(n) > 0: no=open(proot+'.nrd','w')
 
  for line in of[0]: 
    tmparr=line.split()
    min_chi=float(tmparr[ilchi[0]])
    min_oline=line
    print(min_chi,min_oline)
    min_mline=mf[0].readline()
    if len(n) > 0: min_nline=nf[0].readline()
    for i in range(len(o)-1):
      oline=of[i+1].readline()
      mline=mf[i+1].readline()
      if len(n) > 0: nline=nf[i+1].readline()
      tmparr=oline.split()
      #print(len(tmparr))
      #print(tmparr)
      #print(i,ilchi[i+1],len(tmparr))
      #print(i,float(tmparr[ilchi[i+1]]))
      if float(tmparr[ilchi[i+1]]) < min_chi and float(tmparr[iteff[i+1]]) > llimit[i+1]*1.01: 
        min_chi=float(tmparr[ilchi[i+1]])
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

  print(path,rvpath)
  print('path,sdir,pixel=',path,sdir,pixel)

  infiles=os.listdir(os.path.join(path,sdir,pixel))  
  datafiles=[]
  zbestfiles=[]
  for ff in infiles: #add subdirs, which may contain zbest files for SDSS/BOSS
    if os.path.isdir(os.path.join(path,sdir,pixel,ff)): 
      extrafiles=os.listdir(os.path.join(path,sdir,pixel,ff))
      for ff2 in extrafiles: 
        infiles.append(os.path.join(ff,ff2))

  print(path,rvpath)
  if rvpath != path:
    print('hola')
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
    if (filename.find('spectra-') > -1 and filename.find('.fits') > -1):
      datafiles.append(os.path.join(path,sdir,pixel,filename))
    elif (filename.find('zbest-') > -1 and filename.find('.fits') > -1):
      zbestfiles.append(os.path.join(rvpath,sdir,pixel,filename))
# BOSS data
    elif (filename.find('spPlate') > -1 and filename.find('.fits') > -1):
      datafiles.append(os.path.join(path,sdir,pixel,filename))
    elif (filename.find('spZbest') > -1 and filename.find('.fits') > -1):
      zbestfiles.append(os.path.join(rvpath,sdir,pixel,filename))
#  DESI RVSPEC files
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


  f = sorted(glob.glob(input))

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
      if (str(type(hdul1[i])) == "<class 'astropy.io.fits.hdu.table.BinTableHDU'>"): #binary tables
        hdu = fits.BinTableHDU.from_columns(hdul1[i].columns, nrows=nrows)
        hdu.header['EXTNAME'] = hdul1[i].header['EXTNAME']
        if (str(type(hdul2[i])) != "<class 'astropy.io.fits.hdu.table.BinTableHDU'>"): 
          print(i, str(type(hdul1[i])),str(type(hdul2[i])))
          print('Warning: the extension ',i, 'in file ',entry,' is not a binary table as expected based on the preceding files. The extension is skipped.')
        for colname in hdul1[i].columns.names:
          if colname in hdul2[i].columns.names:
            hdu.data[colname][nrows1:] = hdul2[i].data[colname]
          else: print('Warning: the file ',entry,' does not include column ',colname,' in extension ',i,' -- ',hdu.header['EXTNAME'])


      elif (str(type(hdul1[i])) == "<class 'astropy.io.fits.hdu.image.ImageHDU'>"): #images
        hdu = fits.PrimaryHDU(vstack( (hdul1[i].data, hdul2[i].data) ))
        hdu.header['EXTNAME'] = hdul1[i].header['EXTNAME']

      if i == 1: 
        hdu1 = hdu 
      else: 
        hdu2 = hdu 

    hdul1 = fits.HDUList([hdu0,hdu1,hdu2])

  hdul1.writeto(output)

  return(None)


#process a single pixel
def do(path,pixel,sdir='',truth=None,nthreads=1,rvpath=None, libpath='.', config='desi-n.yaml'):
  
  #get input data files
  datafiles,zbestfiles  = finddatafiles(path,pixel,sdir,rvpath=rvpath) 
  if (datafiles == None or zbestfiles == None): return None

  #identify data source
  datafile=datafiles[0]
  hdu=fits.open(datafile)
  enames=extnames(hdu)
  if 'FIBERMAP' in enames: 
    source='desi'
  else:
    source='boss'

  #gather config. info
  ydir = os.path.dirname(os.path.realpath(__file__))
  yfile=open(os.path.join(ydir,config),'r')
  conf=yaml.full_load(yfile)
  yfile.close()
  #set the set of grids to be used
  grids=conf['grids']
  print('grids=',grids)
  bands=conf['bands']


  #loop over possible multiple data files in the same pixel
  for ifi in range(len(datafiles)):

    datafile=datafiles[ifi]
    zbestfile=zbestfiles[ifi]
    fileroot=datafile.split('.')[-2].split('/')[-1]
    print('datafile=',datafile)
    print('fileroot=',fileroot)

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

    if source == 'desi': #DESI data
      fibermap=hdu['FIBERMAP']
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
      

    else:  #SDSS/BOSS data

      plate=pheader['PLATEID']
      mjd=pheader['MJD']
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
      x1=lambda_synth(os.path.join(libpath,gridfile))

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
      k=0
      print('nspec,len(z),npass,len(x1)=',nspec,len(z),npass,len(x1))
      for i in range(nspec):
        if process_target[i]:
          y2[k,:]=interp(x1,x*(1.-z[targetid[i]]),y[i,:])
          ey2[k,:]=interp(x1,x*(1-z[targetid[i]]),ey[i,:])
          k=k+1

      if (j==0):
        xx=x1
        yy=y2
        eyy=ey2
      else:
        xx=concatenate((xx,x1))
        yy=concatenate((yy,y2),axis=1)
        eyy=concatenate((eyy,ey2),axis=1)

      savetxt(os.path.join(sdir,pixel,fileroot)+'-'+bands[j]+'.wav',x1,fmt='%14.5e')

    savetxt(os.path.join(sdir,pixel,fileroot)+'.wav',xx,fmt='%14.5e')
    fmp = tbl.Table(fibermap.data) [process_target]
    hdu0 = fits.BinTableHDU(fmp)
    hdu0.writeto(os.path.join(sdir,pixel,fileroot)+'.fmp.fits')

    write_ferre_input(fileroot,ids,par,yy,eyy,path=os.path.join(sdir,pixel))

    #write slurm script
    write_slurm(fileroot,path=os.path.join(sdir,pixel),
            ngrids=len(grids),nthreads=nthreads, config=config)


    #loop over all grids
    mknml(conf,fileroot,nthreads=nthreads,libpath=libpath,path=os.path.join(sdir,pixel))

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

def main(args):

  parser = argparse.ArgumentParser(description='prepare a data set for processing with FERRE')

  parser.add_argument('-p','--path',
                      type=str,
                      help='path to the input spectra dir tree',
                      default=None)

  parser.add_argument('-t','--truthfile',
                      type=str,
                      help='truth file for DESI simulations',
                      default=None)

  parser.add_argument('-rv','--rvpath',
                      type=str,
                      help='path to the RV input data',
                      default=None)

  parser.add_argument('-l','--libpath',
                      type=str,
                      help='path to the libraries, if not in the current dir',
                      default='.')

  parser.add_argument('-n','--nthreads',
                      type=int,
                      help='number of threads per FERRE job',
                      default=4)
                      
  parser.add_argument('-c','--config',
                      type=str,
                      help='yaml configuration file for FERRE runs',
                      default='desi-n.yaml')

  args = parser.parse_args()

  path=args.path
  rvpath=args.rvpath
  if rvpath is None: rvpath=path

  truthfile=args.truthfile
  if (truthfile is not None):  
    truthtuple=readtruth(truthfile)
  else: truthtuple=None

  config=args.config
  libpath=args.libpath
  nthreads=args.nthreads

  pixels=getpixels(path)
  
  for entry in pixels:
    head, pixel = os.path.split(entry)
    print('head/pixel=',head,pixel)
    sdir=''
    print('path=',path)
    if head != path:
      head, sdir = os.path.split(head)
      if not os.path.exists(sdir): os.mkdir(sdir)
    if sdir != '': 
      if not os.path.exists(sdir):os.mkdir(sdir)
    if not os.path.exists(os.path.join(sdir,pixel)): 
      os.mkdir(os.path.join(sdir,pixel))

    do(path,pixel,sdir=sdir,truth=truthtuple, 
       rvpath=rvpath, libpath=libpath, 
       nthreads=nthreads, config=config,)

    #run(pixel,path=os.path.join(sdir,pixel))
  
if __name__ == "__main__":
  main(sys.argv[1:])
