{\rtf1\ansi\ansicpg1252\cocoartf1265\cocoasubrtf210
{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 Menlo-Regular;\f2\fmodern\fcharset0 Courier;
}
{\colortbl;\red255\green255\blue255;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 setwd("~
\f1\fs22 \CocoaLigature0 /Documents/SocialNetworks/ClickAnalysis/buzzfeed/compare_to_other_methods"
\f0\fs24 \CocoaLigature1 )\
\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 library(ggplot2)\
library("gridExtra")\
require(reshape)\
library(plyr)\
library(RColorBrewer)\
library("cowplot")\
\
rf <- colorRampPalette(rev(brewer.pal(11,'Spectral')))\
r <- rf(32)\
\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 \
data = read.table("
\f1\fs22 \CocoaLigature0 szabopintotransform.txt
\f0\fs24 \CocoaLigature1 ", header=TRUE, sep="\\t")\
\
plot_metric_szabovspinto = function(data) \{\
	d1 = data[data$cps_calc=="-",]\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 	d = melt(d1, id=c("method", "cps_calc", "features", "to_predict"))\
	p = ggplot(d) + geom_point(aes(x=method, y=value, colour =method))\
	p = p  + facet_wrap(~variable, scales="free") \
	p = p + 
\f2 theme(axis.text.x = element_text(angle = 45))
\f0 \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 	return(p)\
\}\
\
\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 plot_metric_transform = function(data) \{\
	d1 = data[data$cps_calc!="-",]\
	d = melt(d1, id=c("method", "cps_calc", "features", "to_predict"))\
	d$g = paste(d$method, d$to_predict, sep="_")\
	d$cps_calc = factor(d$cps_calc, levels=c("-", "CPS=hour1_est_clicks/hour1_est_retweets",  "CPS=hour24_est_clicks/hour24_est_retweets" , "CPS=hour1_actual_clicks/hour1_est_retweets" ,"CPS=hour24_actual_clicks/hour24_est_retweets", "CPS=total_clicks/hour24_est_retweets"  ))\
	p = ggplot(d) + geom_point(aes(x=cps_calc, y=value, colour=to_predict, shape =method)) + geom_line(aes(x= cps_calc, y=value, colour=to_predict, linetype=method, group=g))\
	p = p + facet_wrap(~variable, scales="free")\
	p = p + 
\f2 theme(axis.text.x = element_text(angle = 45, size=7, hjust=1), legend.position="bottom", legend.direction="vertical")
\f0 \
	return(p)\
\}\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 \
# szabo vs pinto\
#p1 = plot_metric_szabovspinto(data, "MAE")\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 #p2 = plot_metric_szabovspinto(data, "MSE")\
#p3 = plot_metric_szabovspinto(data, "medianAE")\
#p4 = plot_metric_szabovspinto(data, "R2")\
#p5 = plot_metric_szabovspinto(data, "MAPE")\
#p = plot_grid(p1, p2, p3, p4, p5, nrow=2, ncol=3)\
p = plot_metric_szabovspinto(data)\
ggsave("szabo_vs_pinto_comparison.png", plot=p, width=10, height=6)\
\
\
p = plot_metric_transform(data)\
ggsave("transforms_comparison.png", plot=p, width=8, height=8)\
\
}