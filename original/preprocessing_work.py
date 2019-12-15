# -*- coding: utf-8 -*-

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20181214"

"""
This is the original program for data preprocessing for my graduation thesis in Finance Dep. WHU.
It fill the nan to 0 and normalized all the data.
You have to change some params by hand. 
Do not use this one; but the codes are correct. 
"""
#--* import pakages *--#
import numpy as np
import pandas as pd
from sklearn import preprocessing
import datetime

#--* define para class*--#
class Para:
    path_data='.\\csv_demo\\'
    path_results='.\\results_demo\\'
para=Para()

starttime = datetime.datetime.now()
#--* define multifactor model class*--#
class _preprocessing_:
    def __init__(self):
        return
    def factor_read(self):

        size=pd.read_csv('1_size.csv',low_memory=False)
        size_ia=pd.read_csv('2_size_ia.csv',low_memory=False)
        beta=pd.read_csv('3_beta.csv',low_memory=False)
        betasq=pd.read_csv('4_betasq.csv',low_memory=False)
        betad=pd.read_csv('5_betad.csv',low_memory=False)
        idvol=pd.read_csv('6_idvol.csv',low_memory=False)
        vol=pd.read_csv('7_vol.csv',low_memory=False)
        idskew=pd.read_csv('8_idskew.csv',low_memory=False)
        skew=pd.read_csv('9_skewness.csv',low_memory=False)
        coskew=pd.read_csv('10_coskew.csv',low_memory=False)
        turn=pd.read_csv('11_turn.csv',low_memory=False)
        std_turn=pd.read_csv('12_std_vol.csv',low_memory=False)
        volumed=pd.read_csv('13_volumed.csv',low_memory=False)
        std_dvol=pd.read_csv('14_std_dvol.csv',low_memory=False)
        retnmax=pd.read_csv('15_retmax.csv',low_memory=False)
        illq=pd.read_csv('16_illq.csv',low_memory=False)
        LM=pd.read_csv('17_LM.csv',low_memory=False)
        sharechg=pd.read_csv('18_sharechg.csv',low_memory=False)
        age=pd.read_csv('19_age.csv',low_memory=False)
        aeavol=pd.read_csv('20_aeavol.csv',low_memory=False)
        #baspres=pd.read_csv('21_baspread.csv')
        pricedelay=pd.read_csv('22_pricedelay.csv',low_memory=False)
        #IPO=pd.read_csv('23_IPO.csv')
        mom12=pd.read_csv('24_mom12.csv',low_memory=False)
        mom6=pd.read_csv('25_mom6.csv',low_memory=False)
        mom36=pd.read_csv('26_mom36.csv',low_memory=False)
        momchg=pd.read_csv('27_momchg.csv',low_memory=False)
        imom=pd.read_csv('28_imom.csv',low_memory=False)
        lagretn=pd.read_csv('29_lagretn.csv',low_memory=False)
        BM=pd.read_csv('30_BM.csv',low_memory=False)
        BM_ia=pd.read_csv('31_BM_ia.csv',low_memory=False)
        AM=pd.read_csv('32_AM.csv',low_memory=False)
        LEV=pd.read_csv('33_LEV.csv',low_memory=False)
        EP=pd.read_csv('34_EP.csv',low_memory=False)
        CFP=pd.read_csv('35_CFP.csv',low_memory=False)###################
        CFP_ia=pd.read_csv('36_CFP_ia.csv',low_memory=False)##################
        OCFP=pd.read_csv('37_OCFP.csv',low_memory=False)######################
        DP=pd.read_csv('38_DP.csv',low_memory=False)
        SP=pd.read_csv('39_SP.csv',low_memory=False)
        AG=pd.read_csv('41_AG.csv',low_memory=False)
        LG=pd.read_csv('42_LG.csv',low_memory=False)
        BVEG=pd.read_csv('43_BVEG.csv',low_memory=False)
        SG=pd.read_csv('44_SG.csv',low_memory=False)
        PMG=pd.read_csv('45_PMG.csv',low_memory=False)
        INVG=pd.read_csv('46_INVG.csv',low_memory=False)
        INVchg=pd.read_csv('47_INVchg.csv',low_memory=False)
        SGINVG=pd.read_csv('48_SGINVG.csv',low_memory=False)
        TAXchg=pd.read_csv('49_TAXchg.csv',low_memory=False)
        acc=pd.read_csv('50_ACC.csv',low_memory=False)####################
        absacc=acc.abs()####################
        stdacc=pd.read_csv('52_stdacc.csv',low_memory=False)####################
        ACCP=pd.read_csv('53_ACCP.csv',low_memory=False)####################
        cinvest=pd.read_csv('54_cinvest.csv',low_memory=False)####################
        depr=pd.read_csv('55_depr.csv',low_memory=False)####################
        pchdepr=pd.read_csv('56_pchdepr.csv',low_memory=False)####################
        egr=pd.read_csv('57_egr.csv',low_memory=False)
        fgr5yr=pd.read_csv('58_fgr5yr.csv',low_memory=False)####################
        grCAPX=pd.read_csv('59_grCAPX.csv',low_memory=False)####################
        pchcapx_ia=pd.read_csv('60_pchcapx_ia.csv',low_memory=False)####################
        #ID=pchcapx_ia.pop('ID')####################
        grithnoa=pd.read_csv('61_grithnoa.csv',low_memory=False)
        invest=pd.read_csv('62_invest.csv',low_memory=False)####################
        pchsale_pchinvt=pd.read_csv('63_pchsale_pchinvt.csv',low_memory=False)
        pchsale_pchrect=pd.read_csv('64_pchsale_pchrect.csv',low_memory=False)
        pchsale_pchxsga=pd.read_csv('65_pchsale_pchxsga.csv',low_memory=False)
        realestate=pd.read_csv('66_realestate.csv',low_memory=False)
        sgr=pd.read_csv('67_sgr.csv',low_memory=False)
        NOA=pd.read_csv('68_NOA.csv')
        hire=pd.read_csv('69_hire.csv',low_memory=False)####################
        chepm_ia=pd.read_csv('70_chempia.csv',low_memory=False)####################
        #ID2=chepm_ia.pop('ID')
        ROE=pd.read_csv('71_ROE.csv',low_memory=False)
        ROA=pd.read_csv('72_ROA.csv',low_memory=False)
        CT=pd.read_csv('73_CT.csv',low_memory=False)
        PA=pd.read_csv('74_PA.csv',low_memory=False)
        cashpr=pd.read_csv('75_cashpr.csv',low_memory=False)
        cash=pd.read_csv('76_cash.csv',low_memory=False)
        RD=pd.read_csv('77_RD.csv',low_memory=False)
        rdi=pd.read_csv('78_rdi.csv',low_memory=False)
        rd_mve=pd.read_csv('79_rd_mve.csv',low_memory=False)
        RDsale=pd.read_csv('80_RDsale.csv',low_memory=False)
        operprof=pd.read_csv('81_operprof.csv',low_memory=False)
        pchgm_pchsale=pd.read_csv('82_pchgm_pchsale.csv',low_memory=False)
        ATO=pd.read_csv('83_ATO.csv',low_memory=False)
        #chfeps=pd.read_excel('84_chfeps.xlsx')#####################
        #85
        nincr=pd.read_csv('86_nincr.csv',low_memory=False)
        #87
        #88
        roic=pd.read_csv('89_roic.csv',low_memory=False)
        rusp=pd.read_csv('90_rusp.csv',low_memory=False)
        #91
        #sue=pd.read_excel('92_sue.xlsx')
        #sfe=pd.read_excel('93_sfe.xlsx')#############
        CR=pd.read_csv('94_CR.csv',low_memory=False)
        QR=pd.read_csv('95_QR.csv',low_memory=False)
        CFdedt=pd.read_csv('96_CFdebt.csv',low_memory=False)
        salecash=pd.read_csv('97_salecash.csv',low_memory=False)
        saleinv=pd.read_csv('98_saleinv.csv',low_memory=False)
        CRG=pd.read_csv('99_CRG.csv',low_memory=False)
        QRG=pd.read_csv('100_QRG.csv',low_memory=False)
        pchsaleinv=pd.read_csv('101_pchsaleinv.csv',low_memory=False)
        salerec=pd.read_csv('102_salerec.csv',low_memory=False)
        #103
        #104
        tang=pd.read_csv('105_tang.csv',low_memory=False)
        #chnanalyst=pd.read_excel('106_chnanalyst.xlsx')#######################
        #nanalyst=pd.read_excel('107_nanalyst.xlsx')######################
        divi=pd.read_csv('108_divi.csv',low_memory=False)
        divo=pd.read_csv('109_divo.csv',low_memory=False)
        herf=pd.read_csv('110_herf.csv',low_memory=False)
        sin=pd.read_csv('111_sin.csv',low_memory=False)
        ret=pd.read_excel('risk_premium.xlsx')
        stockid = pd.read_csv('stockid.csv',low_memory=False)
        industryid=pd.read_csv('industryid.csv',low_memory=False)
        stockid=stockid['stockid']
        industryid=industryid['industryid']
        ret=ret.iloc[:,1:]
        size=size.iloc[:,1:]
        size_ia=size_ia.iloc[:,1:]
        beta=beta.iloc[:,1:]
        betasq=betasq.iloc[:,1:]
        betad=betad.iloc[:,1:]
        idvol=idvol.iloc[:,1:]
        vol=vol.iloc[:,1:]
        idskew=idskew.iloc[:,1:]
        skew=skew.iloc[:,1:]
        coskew=coskew.iloc[:,1:]
        turn=turn.iloc[:,1:]
        std_turn=std_turn.iloc[:,1:]
        volumed=volumed.iloc[:,1:]
        std_dvol=std_dvol.iloc[:,1:]
        retnmax=retnmax.iloc[:,1:]
        illq=illq.iloc[:,1:]
        LM=LM.iloc[:,1:]
        sharechg=sharechg.iloc[:,1:]
        age=age.iloc[:,1:]
        aeavol=aeavol.iloc[:,1:]
        pricedelay=pricedelay.iloc[:,1:]
        mom12=mom12.iloc[:,1:]
        mom6=mom6.iloc[:,1:]
        mom36=mom36.iloc[:,1:]
        momchg=momchg.iloc[:,1:]
        imom=imom.iloc[:,1:]
        lagretn=lagretn.iloc[:,1:]
        BM=BM.iloc[:,1:]
        BM_ia=BM_ia.iloc[:,1:]
        AM=AM.iloc[:,1:]
        LEV=LEV.iloc[:,1:]
        EP=EP.iloc[:,1:]
        CFP=CFP.iloc[:,1:]#################
        CFP_ia=CFP_ia.iloc[:,1:]#################
        OCFP=OCFP.iloc[:,1:]#################
        DP=DP.iloc[:,1:]
        SP=SP.iloc[:,1:]
        AG=AG.iloc[:,1:]
        LG=LG.iloc[:,1:]
        BVEG=BVEG.iloc[:,1:]
        SG=SG.iloc[:,1:]
        PMG=PMG.iloc[:,1:]
        INVG=INVG.iloc[:,1:]
        INVchg=INVchg.iloc[:,1:]
        SgINVg=SGINVG.iloc[:,1:]
        TAXchg=TAXchg.iloc[:,1:]#################
        ACC=acc.iloc[:,1:]#################
        absacc=absacc.iloc[:,1:]#################
        stdacc=stdacc.iloc[:,1:]#################
        ACCP=ACCP.iloc[:,1:]#################
        cinvest=cinvest.iloc[:,1:]#################
        depr=depr.iloc[:,1:]#################
        pchdepr=pchdepr.iloc[:,1:]#################
        egr=egr.iloc[:,1:]
        fgr5yr=fgr5yr.iloc[:,1:]#################
        grCAPX=grCAPX.iloc[:,1:]#################
        pchcapx_ia=pchcapx_ia.iloc[:,1:]#################
        grithnoa=grithnoa.iloc[:,1:]
        invest=invest.iloc[:,1:]#################
        pchsale_pchinvt=pchsale_pchinvt.iloc[:,1:]
        pchsale_pchrect=pchsale_pchrect.iloc[:,1:]
        pchsale_pchxsga=pchsale_pchxsga.iloc[:,1:]
        realestate=realestate.iloc[:,1:]
        sgr=sgr.iloc[:,1:]
        NOA=NOA.iloc[:,1:]
        hire=hire.iloc[:,1:]#################
        chepm_ia=chepm_ia.iloc[:,1:]#################
        ROE=ROE.iloc[:,1:]
        ROA=ROA.iloc[:,1:]
        CT=CT.iloc[:,1:]
        PA=PA.iloc[:,1:]
        cashpr=cashpr.iloc[:,1:]
        cash=cash.iloc[:,1:]
        RD=RD.iloc[:,1:]
        rdi = rdi.iloc[:,1:]
        rd_mve=rd_mve.iloc[:,1:]
        RDsale=RDsale.iloc[:,1:]
        operprof=operprof.iloc[:,1:]
        pchgm_pchsale=pchgm_pchsale.iloc[:,1:]
        ATO=ATO.iloc[:,1:]
        #=chfeps.iloc[:,1:]#################
        nincr=nincr.iloc[:,1:]
        roic=roic.iloc[:,1:]
        rusp=rusp.iloc[:,1:]
        ##sfe=sfe.iloc[:,1:]#################
        CR=CR.iloc[:,1:]
        QR=QR.iloc[:,1:]
        CFdebt=CFdedt.iloc[:,1:]#################
        salecash=salecash.iloc[:,1:]
        saleinv=saleinv.iloc[:,1:]
        CRG=CRG.iloc[:,1:]
        QRG=QRG.iloc[:,1:]
        pchsaleinv=pchsaleinv.iloc[:,1:]
        salerec=salerec.iloc[:,1:]
        tang=tang.iloc[:,1:]
        #chnanalyst=chnanalyst.iloc[:,1:]#################
        #nanalyst=nanalyst.iloc[:,1:]#################
        divi=divi.iloc[:,1:]
        divo=divo.iloc[:,1:]
        herf=herf.iloc[:,1:]
        sin=sin.iloc[:,1:]

        size=np.array(size)
        size_ia=np.array(size_ia)
        beta=np.array(beta)
        betasq=np.array(betasq)
        betad=np.array(betad)
        idvol=np.array(idvol)
        vol=np.array(vol)
        idskew=np.array(idskew)
        skew=np.array(skew)
        coskew=np.array(coskew)
        turn=np.array(turn)
        std_turn=np.array(std_turn)
        volumed=np.array(volumed)
        std_dvol=np.array(std_dvol)
        retnmax=np.array(retnmax)
        illq=np.array(illq)
        LM=np.array(LM)
        sharechg=np.array(sharechg)
        age=np.array(age)
        aeavol=np.array(aeavol)
        pricedelay=np.array(pricedelay)
        mom12=np.array(mom12)
        mom6=np.array(mom6)
        mom36=np.array(mom36)
        momchg=np.array(momchg)
        imom=np.array(imom)
        lagretn=np.array(lagretn)
        BM=np.array(BM)
        BM_ia=np.array(BM_ia)
        AM=np.array(AM)
        LEV=np.array(LEV)
        EP=np.array(EP)
        CFP=np.array(CFP)#################
        CFP_ia=np.array(CFP_ia)#################
        OCFP=np.array(OCFP)#################
        #DP=np.array(DP)
        SP=np.array(SP)
        AG=np.array(AG)
        LG=np.array(LG)
        BVEG=np.array(BVEG)
        SG=np.array(SG)
        PMG=np.array(PMG)
        INVG=np.array(INVG)
        INVchg=np.array(INVchg)
        SgINVg=np.array(SGINVG)
        TAXchg=np.array(TAXchg)#################
        ACC=np.array(acc)#################
        absacc=np.array(absacc)#################
        stdacc=np.array(stdacc)#################
        ACCP=np.array(ACCP)#################
        cinvest=np.array(cinvest)#################
        depr=np.array(depr)#################
        pchdepr=np.array(pchdepr)#################
        egr=np.array(egr)
        fgr5yr=np.array(fgr5yr)#################
        grCAPX=np.array(grCAPX)#################
        pchcapx_ia=np.array(pchcapx_ia)#################
        grithnoa=np.array(grithnoa)
        invest=np.array(invest)#################
        pchsale_pchinvt=np.array(pchsale_pchinvt)
        pchsale_pchrect=np.array(pchsale_pchrect)
        pchsale_pchxsga=np.array(pchsale_pchxsga)
        realestate=np.array(realestate)
        sgr=np.array(sgr)
        NOA=np.array(NOA)
        hire=np.array(hire)#################
        chepm_ia=np.array(chepm_ia)#################
        ROE=np.array(ROE)
        ROA=np.array(ROA)
        CT=np.array(CT)
        PA=np.array(PA)
        cashpr=np.array(cashpr)
        cash=np.array(cash)
        RD=np.array(RD)
        rd_mve=np.array(rd_mve)
        RDsale=np.array(RDsale)
        operprof=np.array(operprof)
        pchgm_pchsale=np.array(pchgm_pchsale)
        ATO=np.array(ATO)
        #chfeps=np.array(chfeps)#################
        nincr=np.array(nincr)
        #roic=np.array(roic)
        rusp=np.array(rusp)
        #sfe=np.array(sfe)#################
        CR=np.array(CR)
        QR=np.array(QR)
        CFdebt=np.array(CFdedt)#################
        salecash=np.array(salecash)
        saleinv=np.array(saleinv)
        CRG=np.array(CRG)
        QRG=np.array(QRG)
        pchsaleinv=np.array(pchsaleinv)
        salerec=np.array(salerec)
        tang=np.array(tang)
        #chnanalyst=np.array(chnanalyst)#################
        #nanalyst=np.array(nanalyst)#################
        #divi=np.array(divi)
        #divo=np.array(divo)
        #herf=np.array(herf)
        #sin=np.array(sin)
        quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',
                                                                 random_state=0)
        size=quantile_transformer.fit_transform(size)
        size_ia=quantile_transformer.fit_transform(size_ia)
        beta=quantile_transformer.fit_transform(beta)
        betasq=quantile_transformer.fit_transform(betasq)
        betad=quantile_transformer.fit_transform(betad)
        idvol=quantile_transformer.fit_transform(idvol)
        vol=quantile_transformer.fit_transform(vol)
        idskew=quantile_transformer.fit_transform(idskew)
        skew=quantile_transformer.fit_transform(skew)
        coskew=quantile_transformer.fit_transform(coskew)
        turn=quantile_transformer.fit_transform(turn)
        std_turn=quantile_transformer.fit_transform(std_turn)
        volumed=quantile_transformer.fit_transform(volumed)
        std_dvol=quantile_transformer.fit_transform(std_dvol)
        retnmax=quantile_transformer.fit_transform(retnmax)
        illq=quantile_transformer.fit_transform(illq)
        LM=quantile_transformer.fit_transform(LM)
        sharechg=quantile_transformer.fit_transform(sharechg)
        age=quantile_transformer.fit_transform(age)
        aeavol=quantile_transformer.fit_transform(aeavol)
        pricedelay=quantile_transformer.fit_transform(pricedelay)
        mom12=quantile_transformer.fit_transform(mom12)
        mom6=quantile_transformer.fit_transform(mom6)
        mom36=quantile_transformer.fit_transform(mom36)
        momchg=quantile_transformer.fit_transform(momchg)
        imom=quantile_transformer.fit_transform(imom)
        lagretn=quantile_transformer.fit_transform(lagretn)
        BM=quantile_transformer.fit_transform(BM)
        BM_ia=quantile_transformer.fit_transform(BM_ia)
        AM=quantile_transformer.fit_transform(AM)
        LEV=quantile_transformer.fit_transform(LEV)
        EP=quantile_transformer.fit_transform(EP)
        CFP=quantile_transformer.fit_transform(CFP)#################
        CFP_ia=quantile_transformer.fit_transform(CFP_ia)#################
        OCFP=quantile_transformer.fit_transform(OCFP)#################
        #DP=quantile_transformer.fit_transform(DP)
        SP=quantile_transformer.fit_transform(SP)
        AG=quantile_transformer.fit_transform(AG)
        LG=quantile_transformer.fit_transform(LG)
        BVEG=quantile_transformer.fit_transform(BVEG)
        SG=quantile_transformer.fit_transform(SG)
        PMG=quantile_transformer.fit_transform(PMG)
        INVG=quantile_transformer.fit_transform(INVG)
        INVchg=quantile_transformer.fit_transform(INVchg)
        SgINVg=quantile_transformer.fit_transform(SGINVG)
        TAXchg=quantile_transformer.fit_transform(TAXchg)#################
        ACC=quantile_transformer.fit_transform(acc)#################
        absacc=quantile_transformer.fit_transform(absacc)#################
        stdacc=quantile_transformer.fit_transform(stdacc)#################
        ACCP=quantile_transformer.fit_transform(ACCP)#################
        cinvest=quantile_transformer.fit_transform(cinvest)#################
        depr=quantile_transformer.fit_transform(depr)#################
        pchdepr=quantile_transformer.fit_transform(pchdepr)#################
        egr=quantile_transformer.fit_transform(egr)
        fgr5yr=quantile_transformer.fit_transform(fgr5yr)#################
        grCAPX=quantile_transformer.fit_transform(grCAPX)#################
        pchcapx_ia=quantile_transformer.fit_transform(pchcapx_ia)#################
        grithnoa=quantile_transformer.fit_transform(grithnoa)
        invest=quantile_transformer.fit_transform(invest)#################
        pchsale_pchinvt=quantile_transformer.fit_transform(pchsale_pchinvt)
        pchsale_pchrect=quantile_transformer.fit_transform(pchsale_pchrect)
        pchsale_pchxsga=quantile_transformer.fit_transform(pchsale_pchxsga)
        realestate=quantile_transformer.fit_transform(realestate)
        sgr=quantile_transformer.fit_transform(sgr)
        NOA=quantile_transformer.fit_transform(NOA)
        hire=quantile_transformer.fit_transform(hire)#################
        chepm_ia=quantile_transformer.fit_transform(chepm_ia)#################
        ROE=quantile_transformer.fit_transform(ROE)
        ROA=quantile_transformer.fit_transform(ROA)
        CT=quantile_transformer.fit_transform(CT)
        PA=quantile_transformer.fit_transform(PA)
        cashpr=quantile_transformer.fit_transform(cashpr)
        cash=quantile_transformer.fit_transform(cash)
        RD=quantile_transformer.fit_transform(RD)
        rd_mve=quantile_transformer.fit_transform(rd_mve)
        RDsale=quantile_transformer.fit_transform(RDsale)
        operprof=quantile_transformer.fit_transform(operprof)
        pchgm_pchsale=quantile_transformer.fit_transform(pchgm_pchsale)
        ATO=quantile_transformer.fit_transform(ATO)
        #chfeps=quantile_transformer.fit_transform(chfeps)#################
        nincr=quantile_transformer.fit_transform(nincr)
        #roic=quantile_transformer.fit_transform(roic)
        rusp=quantile_transformer.fit_transform(rusp)
        #sfe=quantile_transformer.fit_transform(sfe)#################
        CR=quantile_transformer.fit_transform(CR)
        QR=quantile_transformer.fit_transform(QR)
        CFdebt=quantile_transformer.fit_transform(CFdedt)#################
        salecash=quantile_transformer.fit_transform(salecash)
        saleinv=quantile_transformer.fit_transform(saleinv)
        CRG=quantile_transformer.fit_transform(CRG)
        QRG=quantile_transformer.fit_transform(QRG)
        pchsaleinv=quantile_transformer.fit_transform(pchsaleinv)
        salerec=quantile_transformer.fit_transform(salerec)
        tang=quantile_transformer.fit_transform(tang)
        #chnanalyst=quantile_transformer.fit_transform(chnanalyst)#################
        #nanalyst=quantile_transformer.fit_transform(nanalyst)#################
        #divi=quantile_transformer.fit_transform(divi)
        #divo=quantile_transformer.fit_transform(divo)
        #herf=quantile_transformer.fit_transform(herf)
        #sin=quantile_transformer.fit_transform(sin)
        size=pd.DataFrame(size)
        size_ia=pd.DataFrame(size_ia)
        beta=pd.DataFrame(beta)
        betasq=pd.DataFrame(betasq)
        betad=pd.DataFrame(betad)
        idvol=pd.DataFrame(idvol)
        vol=pd.DataFrame(vol)
        idskew=pd.DataFrame(idskew)
        skew=pd.DataFrame(skew)
        coskew=pd.DataFrame(coskew)
        turn=pd.DataFrame(turn)
        std_turn=pd.DataFrame(std_turn)
        volumed=pd.DataFrame(volumed)
        std_dvol=pd.DataFrame(std_dvol)
        retnmax=pd.DataFrame(retnmax)
        illq=pd.DataFrame(illq)
        LM=pd.DataFrame(LM)
        sharechg=pd.DataFrame(sharechg)
        age=pd.DataFrame(age)
        aeavol=pd.DataFrame(aeavol)
        pricedelay=pd.DataFrame(pricedelay)
        mom12=pd.DataFrame(mom12)
        mom6=pd.DataFrame(mom6)
        mom36=pd.DataFrame(mom36)
        momchg=pd.DataFrame(momchg)
        imom=pd.DataFrame(imom)
        lagretn=pd.DataFrame(lagretn)
        BM=pd.DataFrame(BM)
        BM_ia=pd.DataFrame(BM_ia)
        AM=pd.DataFrame(AM)
        LEV=pd.DataFrame(LEV)
        EP=pd.DataFrame(EP)
        CFP=pd.DataFrame(CFP)#################
        CFP_ia=pd.DataFrame(CFP_ia)#################
        OCFP=pd.DataFrame(OCFP)#################
        #=pd.DataFrame(DP)
        SP=pd.DataFrame(SP)
        AG=pd.DataFrame(AG)
        LG=pd.DataFrame(LG)
        BVEG=pd.DataFrame(BVEG)
        SG=pd.DataFrame(SG)
        PMG=pd.DataFrame(PMG)
        INVG=pd.DataFrame(INVG)
        INVchg=pd.DataFrame(INVchg)
        SgINVg=pd.DataFrame(SGINVG)
        TAXchg=pd.DataFrame(TAXchg)#################
        ACC=pd.DataFrame(acc)#################
        absacc=pd.DataFrame(absacc)#################
        stdacc=pd.DataFrame(stdacc)#################
        ACCP=pd.DataFrame(ACCP)#################
        cinvest=pd.DataFrame(cinvest)#################
        depr=pd.DataFrame(depr)#################
        pchdepr=pd.DataFrame(pchdepr)#################
        egr=pd.DataFrame(egr)
        fgr5yr=pd.DataFrame(fgr5yr)#################
        grCAPX=pd.DataFrame(grCAPX)#################
        pchcapx_ia=pd.DataFrame(pchcapx_ia)#################
        grithnoa=pd.DataFrame(grithnoa)
        invest=pd.DataFrame(invest)#################
        pchsale_pchinvt=pd.DataFrame(pchsale_pchinvt)
        pchsale_pchrect=pd.DataFrame(pchsale_pchrect)
        pchsale_pchxsga=pd.DataFrame(pchsale_pchxsga)
        realestate=pd.DataFrame(realestate)
        sgr=pd.DataFrame(sgr)
        NOA=pd.DataFrame(NOA)
        hire=pd.DataFrame(hire)#################
        chepm_ia=pd.DataFrame(chepm_ia)#################
        ROE=pd.DataFrame(ROE)
        ROA=pd.DataFrame(ROA)
        CT=pd.DataFrame(CT)
        PA=pd.DataFrame(PA)
        cashpr=pd.DataFrame(cashpr)
        cash=pd.DataFrame(cash)
        RD=pd.DataFrame(RD)
        rd_mve=pd.DataFrame(rd_mve)
        RDsale=pd.DataFrame(RDsale)
        operprof=pd.DataFrame(operprof)
        pchgm_pchsale=pd.DataFrame(pchgm_pchsale)
        ATO=pd.DataFrame(ATO)
        #chfeps=pd.DataFrame(chfeps)#################
        nincr=pd.DataFrame(nincr)
        #roic=pd.DataFrame(roic)
        rusp=pd.DataFrame(rusp)
        #sfe=pd.DataFrame(sfe)#################
        CR=pd.DataFrame(CR)
        QR=pd.DataFrame(QR)
        CFdebt=pd.DataFrame(CFdedt)#################
        salecash=pd.DataFrame(salecash)
        saleinv=pd.DataFrame(saleinv)
        CRG=pd.DataFrame(CRG)
        QRG=pd.DataFrame(QRG)
        pchsaleinv=pd.DataFrame(pchsaleinv)
        salerec=pd.DataFrame(salerec)
        tang=pd.DataFrame(tang)
        #chnanalyst=pd.DataFrame(chnanalyst)#################
        #nanalyst=pd.DataFrame(nanalyst)#################
        #divi=pd.DataFrame(divi)
        #divo=pd.DataFrame(divo)
        #herf=pd.DataFrame(herf)
        #sin=pd.DataFrame(sin)

        factors=['stockid','industryid','ret','size',
                       'size_ia','beta','betasq','betad',
                       'idvol','vol','idskew','skew',
                       'coskew','Turn','Std_turn','Volumed',
                       'std_dvol','retnmax','illq','LM',
                       'sharechg','age','aeavol','pricedelay',
                       'mom12','mom6','mom36','momchg',
                       'imom','lagretn','BM','bm_ia',
                       'AM','LEV','EP','CFP',
                       'CFP_ia','OCFP','DP',
                       'SP','AG','LG','BVEG',
                       'SG','PMG','INVG','INVchg',
                       'SgINVg','TAXchg','ACC','absacc','stdacc',
                       'ACCP','cinvest','depr','pchdepr',
                       'egr','fgr5yr','grCAPX','pchcapx_ia',
                       'grithnoa','invest','pchsale_pchinvt',
                       'pchsale_pchrect','Pchsale_pchxsga','Realestate','Sgr',
                       'NOA','hire','chemp_ia','ROE',
                       'ROA','CT',
                       'PA','cashpr','cash','RD','RDI',
                       'rd_mve','RDsale','operprof','pchgm_pchsale',
                       'ATO','nincr','roic',
                       'rusp','CR',
                       'QR','CFdebt','salecash','saleinv',
                       'CRG','qRG','pchsaleinv','salerec',
                       'tang','divi',
                       'divo','herf','sin']
        for i in range(len(ret.columns) - 1):
            RET = ret.iloc[:, i]
            Size = size.iloc[:, i]
            Size_ia = size_ia.iloc[:, i]
            Beta = beta.iloc[:, i]
            Betasq = betasq.iloc[:, i]
            Betad = betad.iloc[:, i]
            Idvol = idvol.iloc[:, i]
            Vol = vol.iloc[:, i]
            Idskew = idskew.iloc[:, i]
            Skew = skew.iloc[:, i]
            Coskew = coskew.iloc[:, i]
            Turn = turn.iloc[:, i]
            Std_turn = std_turn.iloc[:, i]
            Volumed = volumed.iloc[:, i]
            Std_dvol = std_dvol.iloc[:, i]
            Retnmax = retnmax.iloc[:, i]
            Illq = illq.iloc[:, i]
            lm = LM.iloc[:, i]
            Sharechg = sharechg.iloc[:, i]
            Age = age.iloc[:, i]
            Aeavol = aeavol.iloc[:, i]
            Pricedelay = pricedelay.iloc[:, i]
            Mom12 = mom12.iloc[:, i]
            Mom6 = mom6.iloc[:, i]
            Mom36 = mom36.iloc[:, i]
            Momchg = momchg.iloc[:, i]
            Imom = imom.iloc[:, i]
            Lagretn = lagretn.iloc[:, i]
            bm = BM.iloc[:, i]
            bm_ia = BM_ia.iloc[:, i]
            am = AM.iloc[:, i]
            lev = LEV.iloc[:, i]
            ep = EP.iloc[:, i]
            cFP = CFP.iloc[:, i]  ########
            cFP_ia = CFP_ia.iloc[:, i]  ########
            oCFP = OCFP.iloc[:, i]  ########
            dp = DP.iloc[:, i]
            sp = SP.iloc[:, i]
            ag = AG.iloc[:, i]
            lg = LG.iloc[:, i]
            bVEG = BVEG.iloc[:, i]
            sG = SG.iloc[:, i]
            pMG = PMG.iloc[:, i]
            iNVG = INVG.iloc[:, i]
            iNVchg = INVchg.iloc[:, i]
            sgINVg = SgINVg.iloc[:, i]  ########
            tAXchg = TAXchg.iloc[:, i]
            ACC = acc.iloc[:, i]  ########
            Absacc = absacc.iloc[:, i]  ########
            Stdacc = stdacc.iloc[:, i]  ########
            aCCP = ACCP.iloc[:, i]  ########
            Cinvest = cinvest.iloc[:, i]  ########
            Depr = depr.iloc[:, i]  ########
            Pchdepr = pchdepr.iloc[:, i]  ########
            Egr = egr.iloc[:, i]
            Fgr5yr = fgr5yr.iloc[:, i]  ########
            GRCAPX = grCAPX.iloc[:, i]  ########
            Pchcapx_ia = pchcapx_ia.iloc[:, i]  ########
            Grithnoa = grithnoa.iloc[:, i]
            Invest = invest.iloc[:, i]  ########
            Pchsale_pchinvt = pchsale_pchinvt.iloc[:, i]
            Pchsale_pchrect = pchsale_pchrect.iloc[:, i]
            Pchsale_pchxsga = pchsale_pchxsga.iloc[:, i]
            Realestate = realestate.iloc[:, i]
            Sgr = sgr.iloc[:, i]
            nOA = NOA.iloc[:, i]
            Hire = hire.iloc[:, i]  ########
            Chempia = chepm_ia.iloc[:, i]  ########
            rOE = ROE.iloc[:, i]
            rOA = ROA.iloc[:, i]
            cT = CT.iloc[:, i]
            pA = PA.iloc[:, i]
            Cashpr = cashpr.iloc[:, i]
            Cash = cash.iloc[:, i]
            rD = RD.iloc[:, i]
            RDI = rdi.iloc[:,i]
            Rd_mve = rd_mve.iloc[:, i]
            rDsale = RDsale.iloc[:, i]
            Operprof = operprof.iloc[:, i]
            Pchgm_pchsale = pchgm_pchsale.iloc[:, i]
            aTO = ATO.iloc[:, i]
            Nincr = nincr.iloc[:, i]
            Roic = roic.iloc[:, i]
            Rusp = rusp.iloc[:, i]
            cR = CR.iloc[:, i]
            qR = QR.iloc[:, i]
            cfdebt = CFdebt.iloc[:, i]  ########
            Salecash = salecash.iloc[:, i]
            Saleinv = saleinv.iloc[:, i]
            cRG = CRG.iloc[:, i]
            qRG = QRG.iloc[:, i]
            Pchsaleinv = pchsaleinv.iloc[:, i]
            Salerec = salerec.iloc[:, i]
            Tang = tang.iloc[:, i]
            Divi = divi.iloc[:, i]
            Divo = divo.iloc[:, i]
            Herf = herf.iloc[:, i]
            Sin = sin.iloc[:, i]

            total = pd.concat([stockid, industryid, RET, Size,
                               Size_ia, Beta, Betasq, Betad,
                               Idvol, Vol, Idskew, Skew,
                               Coskew, Turn, Std_turn, Volumed,
                               Std_dvol, Retnmax, Illq, lm,
                               Sharechg, Age, Aeavol, Pricedelay,
                               Mom12, Mom6, Mom36, Momchg,
                               Imom, Lagretn, bm, bm_ia,
                               am, lev, ep, cFP,
                               cFP_ia, oCFP, dp,  # 3
                               sp, ag, lg, bVEG,
                               sG, pMG, iNVG, iNVchg,
                               sgINVg, tAXchg, ACC, Absacc,
                               Stdacc, aCCP, Cinvest, Depr,
                               Pchdepr, Egr, Fgr5yr, GRCAPX,
                               Pchcapx_ia, Grithnoa, Invest, Pchsale_pchinvt,
                               Pchsale_pchrect, Pchsale_pchxsga, Realestate, Sgr,
                               nOA, Hire, Chempia, rOE,
                               rOA, cT,  # 2
                               pA, Cashpr, Cash, rD,RDI,
                               Rd_mve, rDsale, Operprof, Pchgm_pchsale,
                               aTO, Nincr, Roic,
                               Rusp,
                               cR, qR, cfdebt,
                               Salecash, Saleinv,  # 2
                               cRG, qRG, Pchsaleinv, Salerec,
                               Tang,  Divi,
                               Divo, Herf, Sin], axis=1)  # 4
            # total=total.fillna(total.mean())
            total.columns = factors
            total.set_index(["stockid"], inplace=True)
            import os

            if not os.path.exists(para.path_results):
                os.makedirs(para.path_results)
            filename = para.path_results + '%d' % (i + 1) + '.csv'
            total.to_csv(filename)
        return

if __name__ == '__main__':
    _preprocessing_()

# 计算程序运行时间
endtime = datetime.datetime.now()
def timeStr(s):
    if s < 10:
        return '0' + str(s)
    else:
        return str(s)


print("程序开始运行时间：" + timeStr(starttime.hour) + ":" + timeStr(starttime.minute) + ":" + timeStr(starttime.second))
print("程序结束运行时间：" + timeStr(endtime.hour) + ":" + timeStr(endtime.minute) + ":" + timeStr(endtime.second))

runTime = (endtime - starttime).seconds
runTimehour = runTime // 3600  # 除法并向下取整，整除
runTimeminute = (runTime - runTimehour * 3600) // 60
runTimesecond = runTime - runTimehour * 3600 - runTimeminute * 60
print("程序运行耗时："+str(runTimehour)+"时"+str(runTimeminute)+"分"+str(runTimesecond)+"秒")