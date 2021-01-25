#!/home/callende/anaconda3/bin/python3
'''
Interface to use FERRE from python for DESI/BOSS data

use: piferre -sp path-to-spectra [-rv rvpath -l libpath -spt sptype -rvt rvtype -n nthreads -m minutes -c config -t truthfile ]

e.g. piferre -sp /data/spectro/redux/dc17a2/spectra-64 

Author C. Allende Prieto
'''
import pdb
import sys
import os
import glob
import re
import importlib
from numpy import arange,loadtxt,savetxt,zeros,ones,nan,sqrt,interp,concatenate,array,reshape,min,max,where,divide,mean, stack, vstack
from astropy.io import fits
import astropy.table as tbl
import astropy.units as units
import matplotlib.pyplot as plt
import subprocess
import datetime, time
import argparse
import yaml

version = '0.1.0'
clight=299792.458 #km/s

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
def write_slurm(root,nthreads=1,minutes=120,path=None,ngrids=None, 
config='desi-n.yaml'):
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
      f.write("#SBATCH --time="+str(minutes)+"\n") #minutes
      f.write("#SBATCH --ntasks=1" + "\n")
      f.write("#SBATCH --cpus-per-task="+str(nthreads*2)+"\n")
    else:
      f.write("#SBATCH  -J "+str(root)+" \n")
      f.write("#SBATCH  -o "+str(root)+"_%j.out"+" \n")
      f.write("#SBATCH  -e "+str(root)+"_%j.err"+" \n")
      f.write("#SBATCH  -n "+str(nthreads)+" \n")
      hours2 = int(minutes/60) 
      minutes2 = minutes%60
      f.write("#SBATCH  -t "+"{:02d}".format(hours2)+":"+"{:02d}".format(minutes2)+
              ":00"+"\n") #hh:mm:ss
      f.write("#SBATCH  -D "+os.path.abspath(path)+" \n")
    f.write("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# \n")
    f.write("export OMP_NUM_THREADS="+str(nthreads)+"\n")
    f.write("cd "+os.path.abspath(path)+"\n")
    for i in range(ngrids):
      #f.write("cp input.nml-"+root+"_"+str(i)+" input.nml \n")
      f.write("time "+ferre+" -l input.lst-"+root+"_"+str(i)+" >& "+root+".log_"+str(i))
      #if (i == 8): 
      #  f.write( "  \n")
      #else:
      #  f.write( " & \n")
      f.write(" & \n")
    f.write("wait \n")
    f.write("python3 -c \"import sys; sys.path.insert(0, '"+python_path+ \
            "'); from piferre import opfmerge, write_tab_fits, write_mod_fits, cleanup; opfmerge(\'"+\
            str(root)+"\',config='"+config+"\'); write_tab_fits(\'"+\
            str(root)+"\',config='"+config+"\'); write_mod_fits(\'"+\
            str(root)+"\'); cleanup(\'"+\
            str(root)+"\')\"\n")
    f.close()
    os.chmod(os.path.join(path,root+'.slurm'),0o755)

    return None

#remove FERRE I/O files after the final FITS tables have been produced
def cleanup(root):
  vrdfiles = glob.glob(root+'*vrd')
  wavefiles = glob.glob(root+'*wav')
  opffiles = glob.glob(root+'*opf*')
  nmlfiles = glob.glob('input.nml-'+root+'*')
  lstfiles = glob.glob('input.lst-'+root+'*')
  errfiles = glob.glob(root+'*err')
  frdfiles = glob.glob(root+'*frd')
  nrdfiles = glob.glob(root+'*nrd*')
  mdlfiles = glob.glob(root+'*mdl*')
  fmpfiles = glob.glob(root+'*fmp.fits')
  logfiles = glob.glob(root+'.log*')
  slurmfiles = glob.glob(root+'*slurm')
  abufiles = glob.glob(root+'*.?ca?')
  allfiles = vrdfiles + wavefiles + opffiles + nmlfiles + lstfiles + \
              errfiles + frdfiles + nrdfiles + mdlfiles + \
              fmpfiles + logfiles + slurmfiles + abufiles

  print('removing files:',end=' ')
  for entry in allfiles: 
    print(entry,' ')
    os.remove(entry)
  print(' ')

  return
  
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
    n_p=tuple(array(header['N_P'].split(),dtype=int))

    lst=open(os.path.join(path,'input.lst-'+root+'_'+str(k)),'w')
    for run in conf[synth]: #loop over all runs (param + elements)
      #global keywords in yaml adopted first
      nml=dict(conf['global'])
      #adding/override with param keywords in yaml
      if 'param' in conf[synth]:
        for key in conf[synth]['param'].keys(): nml[key]=conf[synth]['param'][key]
      #adding/override with run keywords in yaml
      for key in conf[synth][run].keys(): nml[key]=conf[synth][run][key]
      #check that inter is feasible with this particular grid
      if nml['inter'] in nml:
        if (min(n_p)-1 < inter): nml['inter']=inter
      #from command line
      nml['nthreads']=nthreads
      #from the actual grid
      nml['ndim']=ndim
      for i in range(len(synthfiles)): 
        nml['SYNTHFILE('+str(i+1)+')'] = "'"+os.path.join(libpath,synthfiles[i])+"'"

      #extensions provided in yaml for input/output files are not supplemented with root
      nml['pfile'] = "'"+root+'.'+nml['pfile']+"'"
      nml['ffile'] = "'"+root+'.'+nml['ffile']+"'"
      nml['erfile'] = "'"+root+'.'+nml['erfile']+"'"
      nml['opfile'] = "'"+root+'.'+nml['opfile']+"'"
      nml['offile'] = "'"+root+'.'+nml['offile']+"'"
      nml['sffile'] = "'"+root+'.'+nml['sffile']+"'"

      #get rid of keywords in yaml that are not for the nml file, but for opfmerge or write_tab
      if 'labels' in nml: del nml['labels']
      if 'llimits' in nml: del nml['llimits']
      if 'steps' in nml: del nml['steps']

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

  if filename.find('spectra-') > -1 or filename.find('exp_') > -1 or filename.find('coadd') > -1: #DESI
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

#get dependencies versions, shamelessly copied from rvspec (Koposov's code)
def get_dep_versions():
    """
    Get Packages versions
    """
    packages = [
        'numpy', 'astropy', 'matplotlib', 'scipy',
        'yaml'
    ]
    # Ideally you need to check that the list here matches the requirements.txt
    ret = {}
    for curp in packages:
        ret[curp] = importlib.import_module(curp).__version__
    ret['python'] = str.split(sys.version, ' ')[0]
    return ret


#find out versions 
def get_versions():

  ver = get_dep_versions()
  ver['piferre'] = version
  log0file = glob.glob("*.log_0")
  if len(log0file) < 1:
    print("Warning: cannot find any *.log_0 file in the working directory")
    fversion = 'unknown'
  else:
    l0 = open(log0file[0],'r')
    while 1:
      line = l0.readline()
      if 'f e r r e' in line:
        entries = line.split()
        fversion = entries[-1][1:]
        break
  l0.close()
  ver['ferre'] = fversion

  return(ver)

#write piferre param. output
def write_tab_fits(root, path=None, config='desi-n.yaml'):
  
  if path is None: path=""
  proot=os.path.join(path,root)
  v=glob.glob(proot+".vrd")
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
  rv_adop=[]
  vf=open(v[0],'r')
  of=open(o[0],'r')
  for line in of:
    cells=line.split()
    #for N dim (since COVPRINT=1 in FERRE), there are m= 4 + N*(2+N) cells
    #and likewise we can calculate N = sqrt(m-3)-1
    m=len(cells)
    assert (m > 6), 'Error, the file '+o[0]+' has less than 7 columns, which would correspond to ndim=2'
    ndim=int(sqrt(m-3)-1)
    
    line = vf.readline()
    vcells=line.split()

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
      rv_adop.append(float(vcells[6])*clight)
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
      rv_adop.append(float(vcells[6])*clight)
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
      rv_adop.append(float(vcells[6])*clight)
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
      rv_adop.append(float(vcells[6])*clight)
      snr_med.append(float(cells[10]))
      if (config == 'desi-s.yaml'):
        cov = zeros((3,3))
        cov[:,:] = reshape(array(cells[12:],dtype=float),(4,4))[1:,1:]
        covar.append(cov)    
      else:
        print('Error: this path in the code is not yet ready!')
        #sys.exit()
        cov = zeros((5,5))
        #cov[3:,3:] = reshape(array(cells[8:],dtype=float),(2,2))
        covar.append(cov)    
   

    if (chisq_tot[-1] < 1. and snr_med[-1] > 5.): # chi**2<10 and S/N>5
      success.append(1) 
    else: success.append(0)
    fid.append(cells[0])
    elem.append([nan,nan])
    elem_err.append([nan,nan])


  hdu0=fits.PrimaryHDU()

  #find out processing date and add it to primary header
  now = datetime.datetime.fromtimestamp(time.time())
  nowstr = now.isoformat() 
  nowstr = nowstr[:nowstr.rfind('.')]
  hdu0.header['DATE'] = nowstr

  #get versions and enter then in primary header
  ver = get_versions()
  for entry in ver.keys(): hdu0.header[entry] = ver[entry]
  
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
    pass
    #cols['COVAR'] = array(covar).reshape(len(success),5,5)
  cols['ELEM'] = array(elem)
  cols['ELEM_ERR'] = array(elem_err)
  cols['CHISQ_TOT'] = array(chisq_tot)
  cols['SNR_MED'] = array(snr_med)
  cols['RV_ADOP'] = array(rv_adop)

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
  'SNR_MED': 'Median signal-to-ratio',
  'RV_ADOP': 'Adopted Radial Velocity (km/s)'
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

  #get versions and enter then in primary header
  ver = get_versions()
  for entry in ver.keys(): hdu0.header[entry] = ver[entry]

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

    if i%10 == 0: print('. ',end='',flush=True)
    if i%100 == 0 : print(str(i),end='',flush=True)

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
  print('/')

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
    tmpstr=conf[entry]['param']['llimits']
    tmplist=tmpstr.split()
    llimit.append(float(tmplist[iteffcol]))
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

#find pixels in 'root' directory (spectra-64)
def getpixels(root):
  #root='spectro/redux/dc17a2/spectra-64/'
  d1=os.listdir(root)
  d=[]
  for x in d1:
    print('x=',x)
    assert os.path.isdir(os.path.join(root,x)), 'the data directory must contain folders, not data files'
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

#get spectra and matching RV files
def getdata(sppath='.',rvpath=None,sptype='spectra',rvtype='zbest'):

  if rvpath is None: rvpath = sppath

  spfiles1 = list(glob.iglob(os.path.join(sppath,'**',sptype+'*fits'), recursive=True))
  rvfiles1 = list(glob.iglob(os.path.join(rvpath,'**',rvtype+'*fits'), recursive=True))
  
  print('spfiles1=',spfiles1)
  print('rvfiles1',rvfiles1)

  spfiles = []
  rvfiles = []
  for entry in spfiles1:
    filename = os.path.split(entry)[-1]
    #print('filename=',filename,' pattern=',filename[len(sptype):])
    entry2 = list(filter(lambda x: filename[len(sptype):] in x, rvfiles1))
    if len(entry2) == 1:
      spfiles.append(entry)
      rvfiles.append(entry2[0])
    elif len(entry2) < 1:
      print('Warning: there is no matching rv file for ',entry,' -- we skip this file')
    else:
      print('Warning: there are multiple matching rv files for ',entry,' -- we skip this file')
      print('         rv matching files:',' '.join(entry2))
  
  #analyze the situation wrt input files
  nsp=len(spfiles)
  nrv=len(rvfiles)

  print ('Found '+str(nsp)+' input spectra files')
  for filename in spfiles: print(filename+'--')
  print ('and '+str(nrv)+' associated rv files')
  for filename in rvfiles: print(filename+'--')

  if (nsp != nrv):
    print('ERROR -- there is a mismatch between the number of spectra files and rv files, this pixel is skipped')
    return (None,None)

  return (spfiles,rvfiles)


#identify input data files and associated zbest files 
#obsolete, use getdata instead
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

#inspector
def inspector(*args):

  for file in args:
    print('file=',file,' file[:6]=',file[:5])
    if file[:5] == 'sptab':
      sph=fits.open(file)
      spt=sph['SPTAB'].data
      fbm=sph['FIBERMAP'].data

      sym='.'
      plt.figure(1)

      plt.subplot(2,2,1)
      plt.plot(spt['teff'],spt['logg'],sym)
      plt.xlabel('Teff')
      plt.ylabel('logg')
      plt.title(file)

      plt.xlim([max(spt['teff'])*1.01,min(spt['teff'])*.99])
      plt.ylim([max(spt['logg'])*1.01,min(spt['logg'])*0.99])
      plt.xscale('log')

      plt.subplot(2,2,2)
      plt.plot(fbm['target_ra'],fbm['target_dec'],sym)
      plt.xlabel('target_ra')
      plt.ylabel('target_dec')
      #plt.xlim([max(spt['teff'])]*1.01,min(spt['teff'])*.99])
      #plt.xlim([max(spt['logg'])]*1.01,min(spt['logg'])*0.99])

      plt.subplot(2,2,3)
      plt.plot(spt['teff'],spt['logg'],sym)
      plt.xlabel('Teff')
      plt.ylabel('logg')

      plt.xlim([8000.,min(spt['teff'])*0.99])
      plt.ylim([5.5,-0.5])

      plt.subplot(2,2,4)
      plt.plot(spt['teff'],spt['feh'],sym)
      plt.xlabel('Teff')
      plt.ylabel('[Fe/H]')

      plt.xlim([8000.,min(spt['teff'])*0.99])
      plt.ylim([-5,1])

      plt.show()

  return None

#process a single pixel
def do(path, pixel, sdir='', truth=None, nthreads=1,minutes=120, rvpath=None, 
libpath='.', sptype='spectra', rvtype='zbest', config='desi-n.yaml'):
  
  #get input data files
  #datafiles,zbestfiles  = finddatafiles(path,pixel,sdir,rvpath=rvpath) 

  print('path,rvpath,sdir,pixel=',path,rvpath,sdir,pixel)

  datafiles,zbestfiles  = getdata(sppath=os.path.join(path,sdir,pixel),
				rvpath=os.path.join(rvpath,sdir,pixel),
				sptype=sptype, rvtype=rvtype) 
  				  
  if (datafiles == None or zbestfiles == None or 
      len(datafiles) == 0 or len(zbestfiles) == 0): return None

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
            ngrids=len(grids),nthreads=nthreads, minutes=minutes, config=config)


    #loop over all grids
    mknml(conf,fileroot,nthreads=nthreads,libpath=libpath,path=os.path.join(sdir,pixel))

    #run ferre
    #ferrerun(path=os.path.join(sdir,pixel))

    #opfmerge(pixel,path=os.path.join(sdir,pixel))


  return None


def main(args):

  parser = argparse.ArgumentParser(description='prepare a data set for processing with FERRE')

  parser.add_argument('-sp','--sppath',
                      type=str,
                      help='path to the input spectra dir tree (must contain directories)',
                      default=None)

  parser.add_argument('-rv','--rvpath',
                      type=str,
                      help='path to the RV input data',
                      default=None)

  parser.add_argument('-l','--libpath',
                      type=str,
                      help='path to the libraries, if not in the current dir',
                      default='.')

  parser.add_argument('-spt','--sptype',
                      type=str,
                      help='type of data (spectra, coadd, spPlate)',
                      default='spectra')

  parser.add_argument('-rvt','--rvtype',
                      type=str,
                      help='type of RV data (zbest, rvtab_spectra, rvtab_coadd, spZbest)',
                      default='zbest')

  parser.add_argument('-n','--nthreads',
                      type=int,
                      help='number of threads per FERRE job',
                      default=4)

  parser.add_argument('-m','--minutes',
                      type=int,
                      help='requested CPU time in minutes per FERRE job',
                      default=120)
                      
  parser.add_argument('-c','--config',
                      type=str,
                      help='yaml configuration file for FERRE runs',
                      default='desi-n.yaml')

  parser.add_argument('-t','--truthfile',
                      type=str,
                      help='truth file for DESI simulations',
                      default=None)

  args = parser.parse_args()

  sppath=args.sppath
  rvpath=args.rvpath
  if rvpath is None: rvpath=sppath

  libpath=args.libpath

  sptype=args.sptype
  rvtype=args.rvtype

  config=args.config
  nthreads=args.nthreads
  minutes=args.minutes

  truthfile=args.truthfile
  if (truthfile is not None):  
    truthtuple=readtruth(truthfile)
  else: truthtuple=None


  pixels=getpixels(sppath)
  
  for entry in pixels:
    head, pixel = os.path.split(entry)
    print('head/pixel=',head,pixel)
    sdir=''
    print('sppath=',sppath)
    print('rvpath=',rvpath)
    if head != sppath:
      head, sdir = os.path.split(head)
      if not os.path.exists(sdir): os.mkdir(sdir)
    if sdir != '': 
      if not os.path.exists(sdir):os.mkdir(sdir)
    if not os.path.exists(os.path.join(sdir,pixel)): 
      os.mkdir(os.path.join(sdir,pixel))

    do(sppath,pixel,sdir=sdir,truth=truthtuple, 
       rvpath=rvpath, libpath=libpath, 
       sptype=sptype, rvtype=rvtype,
       nthreads=nthreads, minutes=minutes, config=config)

    #run(pixel,path=os.path.join(sdir,pixel))
  
if __name__ == "__main__":
  main(sys.argv[1:])
