{\rtf1\ansi\ansicpg1252\cocoartf1265\cocoasubrtf210
{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 Menlo-Regular;\f2\fmodern\fcharset0 Courier;
}
{\colortbl;\red255\green255\blue255;}
\margl1440\margr1440\vieww18480\viewh11020\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 setwd("~/Documents/SocialNetworks/
\f1\fs22 \CocoaLigature0 ClickAnalysis
\f0\fs24 \CocoaLigature1 /buzzfeed/")\
library(ggplot2)\
library("gridExtra")\
require(reshape2)\
library(plyr)\
library(RColorBrewer)\
library("cowplot")\
require(reshape)\
\
rf <- colorRampPalette(rev(brewer.pal(11,'Spectral')))\
r <- rf(32)\
\
filename = "
\f1\fs22 \CocoaLigature0 master_temporal_dataset.csv
\f0\fs24 \CocoaLigature1 "\
data = read.table(filename, sep=",", header=TRUE)\
\
# scatter plots of correlations of impressions vs receptions\
p_orig = ggplot(data, aes(x= actual_impressions, y= est_impressions)) + geom_point(aes(colour=factor(hour)))   + geom_line(aes(group=X_id))\
p_orig = p_orig + ylab("Receptions") + xlab("Impressions") + 
\f2 geom_abline(colour="red")
\f0 \
p_orig = p_orig + scale_x_log10() + scale_y_log10()\
\
estimate_geom = function(q, s, col_name, d)\{\
\pard\pardeftab720

\f2 \cf0 	a = d[, c("X_id", "hour", "est_impressions")]\
	a$hour = paste("hr", a$hour, sep="_")\
	names(a) = c("X_id", "variable", "value")\
	b = cast(a)\
\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural

\f0 \cf0 	#z1 = s*(1-q)*x1\
	#z2 = (1-q^2)*s*x1 + s*(x2(1-q))\
	#z3 = s*(1-q^3)*x1+ s*(x2(1-q)^2)+ s*(x3(1-q))\
\pard\pardeftab720

\f2 \cf0 \
	# s = scaling factor\
	# q = 
\f0 q = exp(-lambda*tau) <-- how much it decays
\f2 \
	x1 = 
\f0 b$hr_1\
	x2 = b$hr_2 - b$hr_1
\f2 \

\f0 	x3 = b$hr_3 - b$hr_2
\f2 \

\f0 	x4 = b$hr_4 - b$hr_3
\f2 \

\f0 	x8 = b$hr_8 - b$hr_4
\f2 \

\f0 	x12 = b$hr_12 - b$hr_8
\f2 \

\f0 	x24 = b$hr_24 - b$hr_12\

\f2 \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0 \cf0 	b$fhr_1 = x1*s*(1-q)\
	b$fhr_2 = x1*s*(1-q^2) + x2*s*(1-q)\
	b$fhr_3 = x1*s*(1-q^3) + x2*s*(1-q^2) + x3*s*(1-q) \
	b$fhr_4 = x1*s*(1-q^4) + x2*s*(1-q^3) + x3*s*(1-q^2) + x4*s*(1-q)\
	b$fhr_8 = x1*s*(1-q^8) + x2*s*(1-q^7) + x3*s*(1-q^6) + x4*s*(1-q^5) + x8*s*(1-q^4)\
	b$fhr_12 = x1*s*(1-q^12) + x2*s*(1-q^11) + x3*s*(1-q^10) + x4*s*(1-q^9) + x8*s*(1-q^8) + x12*s*(1-q^4)\
	b$fhr_24 = x1*s*(1-q^24) + x2*s*(1-q^23) + x3*s*(1-q^22) + x4*s*(1-q^21) + x8*s*(1-q^20) + x12*s*(1-q^16) + x24*s*(1-q^12)\
	\
\
	d[d$hour==1, col_name] = b$fhr_1\
	d[d$hour==2, col_name] = b$fhr_2\
	d[d$hour==3, col_name] = b$fhr_3\
	d[d$hour==4, col_name] = b$fhr_4\
	d[d$hour==8, col_name] = b$fhr_8\
	d[d$hour==12, col_name] = b$fhr_12\
	d[d$hour==24, col_name] = b$fhr_24\
\
	scaling_factor = b$fhr_24/d[d$hour==24, "actual_impressions"] # est impressions/receptions\
	print(head(d[,col_name]))\
	d[, col_name] = d[, col_name] / mean(scaling_factor)\
	print(head(d[,col_name]))\
	d[, paste(col_name, "_s", sep="")] = scaling_factor\
	print(summary(scaling_factor))\
	return(d)\
\
\}\
\
make_plot = function(r, col_name)\{\
	p = ggplot(data, aes_string(x= "actual_impressions", y= col_name))  + geom_line(aes(group=X_id))+ geom_point(aes(colour=factor(hour))) \
	p = p + ylab("Estimated Impressions") + xlab("Impressions") + ggtitle(paste("estimating impressions (geometric transform); r=", r))\
	p = p + scale_x_log10() + scale_y_log10() + 
\f2 geom_abline(colour="red")\

\f0 	return(p)	\
\}\
\
data = estimate_geom(0.5, 1, "fhr_geom_r_0_5", data)\
data = estimate_geom(0.75, 1, "fhr_geom_r_0_75", data)\
data = estimate_geom(.9, 1, "fhr_geom_r_0_9", data)\
data = estimate_geom(0.95, 1, "fhr_geom_r_0_95", data)\
data = estimate_geom(0.35, 1, "fhr_geom_r_0_35", data)\
\
p1 = make_plot(.5, "fhr_geom_r_0_5")\
p2 = make_plot(.75, "fhr_geom_r_0_75")\
p4 = make_plot(.35, "fhr_geom_r_0_35")\
p = plot_grid(p_orig, p1, p2, p4, labels=c("original", "r=0.5", "r=0.75", "r=0.9", "r=0.95"), ncol=2, nrow=2)\
ggsave("estimateimpressions_geom_transforms.png", plot=p, width=24, height=20)\
\
\
#---------------------------------------------------------------\
# ERRORS\
\
\
errors = melt(data[, c("X_id", "actual_impressions", "hour",  "fhr_geom_r_0_5", "fhr_geom_r_0_75", "fhr_geom_r_0_9", "fhr_geom_r_0_35")], id=c("X_id", "actual_impressions", "hour"))\
# computing the distance from the correct answer\
errors$error = errors$value-errors$actual_impressions\
# computing the distance from the correct answer - in logscale\
errors$logerror = log(errors$value)-log(errors$actual_impressions)\
errors_summary = ddply(errors, c("hour", "variable"), summarize, mse=mean(error^2)/length(actual_impressions), mse_log=mean(logerror^2)/length(actual_impressions), corr = cor(value, actual_impressions))\
\
#errors_summary = ddply(errors, c("hour", "variable"), summarize, mse=mean(error^2)/length(actual_impressions), mse_log=mean(logerror^2)/length(actual_impressions), corr = cor(value, actual_impressions), mean = mean(actual_impressions), ss_tot=sum((actual_impressions-mean)^2), ss_res=sum((value-actual_impressions)^2), r2=1-ss_res/ss_tot)\
\
p1 = ggplot(errors_summary, aes(x=hour, y=mse)) + geom_line(aes(colour=variable)) + geom_point(aes(colour=variable)) + ggtitle("MSE") + scale_y_log10()\
p2 = ggplot(errors_summary, aes(x=hour, y=mse_log)) + geom_line(aes(colour=variable)) + geom_point(aes(colour=variable)) + ggtitle("MSE in log-scale") + scale_y_log10()\
p3 = ggplot(errors_summary, aes(x=hour, y=corr)) + geom_line(aes(colour=variable)) + geom_point(aes(colour=variable)) + ggtitle("correlation of estimate to actual (all urls in a time period)")\
\
errors_summary = ddply(errors, c("X_id", "variable"), summarize, corr = cor(value, actual_impressions))\
p4 = ggplot(errors_summary, aes(x=variable, y=corr)) + geom_boxplot(aes(colour=variable)) + ggtitle("correlation of estimate to actual (by url)")+ 
\f2 theme(axis.text.x = element_text(angle = 30, hjust = 1))
\f0 \
\
p = plot_grid(p1, p2, p3, p4, ncol=2, nrow=2)\
ggsave("estimateimpressions_geom_transforms_errors.png", plot=p, width=20, height=8)\
\
\
\
#---------------------------------------------------------------\
# ESTIMATE r\
rf <- colorRampPalette(rev(brewer.pal(11,'Spectral')))\
r <- rf(32)\
\
p1 = ggplot(data, aes(x=factor(hour), y = actual_clicks/actual_impressions)) + geom_line(aes(group=X_id)) \
p1 = p1 + scale_y_continuous(trans="log1p",breaks=c(1e-3, 1e-2, 1e-1, 1e0, .25, .5, .75))\
p2 = ggplot(data, aes(x=factor(hour), y = actual_impressions/est_impressions)) + stat_bin2d() \
p2 = p2 + scale_fill_gradientn(colours=r, trans="log", breaks=c(1,1e1,50,1e2,1e3), labels=c("1", "10", "50", "100", "1000"))\
p2 = p2 + scale_y_continuous(trans="log1p",breaks=c(1e-3, 1e-2, 1e-1, 1e0, .25, .5, .75))\
\
p2 = ggplot(data, aes(x=factor(hour), y = actual_impressions/est_impressions)) + stat_bin2d() \
p2 = p2 + scale_fill_gradientn(colours=r, trans="log", breaks=c(1,1e1,50,1e2,1e3), labels=c("1", "10", "50", "100", "1000"))\
p2 = p2 + scale_y_log10()\
\
#p3 = ggplot(data, aes(x=factor(hour), y = actual_clicks/actual_impressions)) + geom_boxplot()\
#p3 = p3 + scale_y_continuous(trans="log1p",breaks=c(1e-3, 1e-2, 1e-1, 1e0, .25, .5, .75))\
\
data = estimate_geom(0.4, 1, "fhr_geom_r_0_4", data)\
data = estimate_geom(0.25, 1, "fhr_geom_r_0_25", data)\
data = estimate_geom(0.35, 1, "fhr_geom_r_0_35", data)\
data = estimate_geom(0.375, 1, "fhr_geom_r_0_375", data)\
\
d = data[, c("X_id", "hour", "est_impressions", "actual_impressions", "fhr_geom_r_0_5", "fhr_geom_r_0_75", "fhr_geom_r_0_4", "fhr_geom_r_0_25", "fhr_geom_r_0_35", "fhr_geom_r_0_375")]\
d$ipr = d$actual_impressions/d$est_impressions\
d$est_ipr_0_5 = d$fhr_geom_r_0_5/d$est_impressions\
#d$est_ipr_0_75 = d$fhr_geom_r_0_75/d$est_impressions\
d$est_ipr_0_4 = d$fhr_geom_r_0_4/d$est_impressions\
#d$est_ipr_0_25 = d$fhr_geom_r_0_25/d$est_impressions\
d$est_ipr_0_35 = d$fhr_geom_r_0_35/d$est_impressions\
d$est_ipr_0_375 = d$fhr_geom_r_0_375/d$est_impressions\
# (1-r)*s*avg_x1 = avg_y1\
b=melt(d, id=c("X_id", "hour", "actual_impressions", "est_impressions", "fhr_geom_r_0_25", "fhr_geom_r_0_4", "fhr_geom_r_0_5", "fhr_geom_r_0_75","fhr_geom_r_0_35", "fhr_geom_r_0_375"))\
\pard\pardeftab720

\f2 \cf0 means <- aggregate(value ~ hour+variable, b, mean)
\f0 \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 p3 = ggplot(b, aes(x=factor(hour), y = value)) + geom_boxplot(aes(colour=variable)) + scale_y_log10()\
\pard\pardeftab720

\f2 \cf0 p3 = p3 + geom_text(data = means, aes(label = formatC(value,2), y = value , colour=variable))\
\
p = plot_grid(p2, p3, labels=c("Impressions/Receptions heatmap", "Impressions/Receptions boxplot"), ncol = 2, nrow = 1)
\f0 \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 ggsave("estimateimpressions_choosing_r.png", plot=p, width=20, height=8)\
\
}