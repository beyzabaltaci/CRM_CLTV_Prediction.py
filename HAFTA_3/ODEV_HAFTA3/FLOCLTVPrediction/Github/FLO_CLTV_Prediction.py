##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO şirketi satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.
###############################################################
# Veri Seti Hikayesi
###############################################################
# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
###############################################################
# 1. Veriyi Hazırlama

import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: ""%.4f" % x)

df_ = pd.read_csv("HAFTA_3/ODEV_HAFTA3/FLOCLTVPrediction/flo_data_20k.csv")
df = df_.copy()
df.head()

# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayalım.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit.round()
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit.round()

df.describe().T
df.dtypes

# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
# aykırı değerleri (varsa) baskılayalım.

for col in df.columns:
    if df[col].dtypes == "float64":
        replace_with_thresholds(df, col)

df.describe().T

# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturalım.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df.head(10)

# 5. Değişken tiplerini inceleyelim. Tarih ifade eden değişkenlerin tipini date'e çevirelim.

df.dtypes

date_cols = [col for col in df.columns if "date" in col]
for col in date_cols:
    df[col] = pd.to_datetime(df[col])

# 3. CLTV Veri Yapısının Oluşturulması
# 3.1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alalım.

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)

# 3.2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturalım.

df_cltv = pd.DataFrame()
df_cltv["customer_id"] = df["master_id"]
df_cltv["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) / 7
df_cltv["T_weekly"] = ((today_date - df["first_order_date"]).astype('timedelta64[D]'))/7
df_cltv["frequency"] = df["order_num_total"]
df_cltv["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]

df_cltv.head()

#Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

# 4. BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
# 4.1. BG/NBD modelini fit edelim.

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(df_cltv["frequency"], df_cltv["recency_cltv_weekly"], df_cltv["T_weekly"])

# a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin edelim ve exp_sales_3_month olarak cltv dataframe'ine ekleyelim.

df_cltv["exp_sales_3_month"] = bgf.predict(4*3,
                                       df_cltv['frequency'],
                                       df_cltv['recency_cltv_weekly'],
                                       df_cltv['T_weekly'])
df_cltv.head()

# b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin edelim ve exp_sales_6_month olarak cltv dataframe'ine ekleyelim.

df_cltv["exp_sales_6_month"] = bgf.predict(4*6,
                                       df_cltv['frequency'],
                                       df_cltv['recency_cltv_weekly'],
                                       df_cltv['T_weekly'])
df_cltv.head()

# 4.2. Gamma-Gamma modelini fit edelim. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyelim.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(df_cltv['frequency'], df_cltv['monetary_cltv_avg'])
df_cltv["exp_average_value"] = ggf.conditional_expected_average_profit(df_cltv['frequency'], df_cltv['monetary_cltv_avg'])
df_cltv.head()

# a. 6 aylık CLTV hesaplayalım ve cltv ismiyle dataframe'e ekleyelim.

df_cltv = ggf.customer_lifetime_value(bgf,
                                   df_cltv['frequency'],
                                   df_cltv['recency_cltv_weekly'],
                                   df_cltv['T_weekly'],
                                   df_cltv['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

# b. Cltv değeri en yüksek 20 kişiyi gözlemleyelim.

df_cltv.sort_values("cltv", ascending=False).head(20)


# 5. CLTV'ye Göre Segmentlerin Oluşturulması
# 5.1. 6 aylık tüm müşterileri 4 gruba (segmente) ayıralım ve grup isimlerini veri setine ekleyelim. cltv_segment ismi ile dataframe'e ekleyelim.

df_cltv["segment"] = pd.qcut(df_cltv["cltv"], 4, labels=["D", "C", "B", "A"])
df_cltv.head()
# 5.2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulunalım.

df_a_b = df_cltv.loc[(df_cltv["segment"] == "A") | (df_cltv["segment"] == "D")]
df_cltv.groupby("segment")["exp_sales_3_month", "exp_sales_6_month", "exp_average_value"].mean()



