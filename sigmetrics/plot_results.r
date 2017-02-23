{\rtf1\ansi\ansicpg1252\cocoartf1265\cocoasubrtf210
{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 Menlo-Regular;\f2\fmodern\fcharset0 Courier;
}
{\colortbl;\red255\green255\blue255;}
\margl1440\margr1440\vieww17740\viewh11220\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 setwd("~/Documents/SocialNetworks/ClickAnalysis/classification/user_features/combined with share features/nytimes")\
library(plyr)\
library(ggplot2)\
require(graphics)\
library(reshape)\
library(colorRamps)\
\
#prefix = "
\f1\fs22 \CocoaLigature0 linearRegression_linRegLogUserShareFeaturesLogClicks
\f0\fs24 \CocoaLigature1 "\
#prefix = "multiclass_svmSVCauto_logUserAndShareFeaturesUnderOverEstimate"\
#prefix = "multiclass_svmSVCauto_logUserAndShareFeaturesUnderOverEstimateNewClasses"\
#data = read.table(paste(prefix, "_results_domain4.txt",sep=""), header=TRUE, sep="\\t")\
\
\
prefix = "linearRegression_linRegLogUserShareFeaturesLogClicksNYTIMES"\
prefix = "linearRegression_linRegLogUserShareFeaturesLogClicksNYTIMESForPpr"\
#prefix="multiclass_svmSVCauto_logUserAndShareFeaturesNYTIMES"\
data = read.table(paste(prefix, "_results_domain3.txt",sep=""), header=TRUE, sep="\\t")\
\
data$diff = data$predicted_value-data$real_value\
\
# for each feature set\
# calculate the total difference, diff (under), diff (over)\
# make a histogram for each\
# make a joint histogram?\
data
\f1\fs22 \CocoaLigature0 $time = sapply(strsplit(as.character(
\f0\fs24 \CocoaLigature1 data
\f1\fs22 \CocoaLigature0 $features), " ", fixed=TRUE), "[", 1)\

\f0\fs24 \CocoaLigature1 data
\f1\fs22 \CocoaLigature0 [!grepl("hr", 
\f0\fs24 \CocoaLigature1 data
\f1\fs22 \CocoaLigature0 $time),"time"] = "-"\

\f0\fs24 \CocoaLigature1 data
\f1\fs22 \CocoaLigature0 $time = factor(
\f0\fs24 \CocoaLigature1 data
\f1\fs22 \CocoaLigature0 $time, levels = c("1hr", "4hr", "24hr", "-"))\

\f0\fs24 \CocoaLigature1 data
\f1\fs22 \CocoaLigature0 $userfeature = sapply(strsplit(as.character(
\f0\fs24 \CocoaLigature1 data
\f1\fs22 \CocoaLigature0 $features), " and ", fixed=TRUE), "[", 2)\

\f0\fs24 \CocoaLigature1 data[
\f1\fs22 \CocoaLigature0 is.na(
\f0\fs24 \CocoaLigature1 data
\f1\fs22 \CocoaLigature0 $userfeature),"userfeature"] = "no_user_features"\

\f0\fs24 \CocoaLigature1 data[!grepl("hr", data$features)
\f1\fs22 \CocoaLigature0 ,"userfeature"] = as.character(
\f0\fs24 \CocoaLigature1 data[!grepl("hr", data$features)
\f1\fs22 \CocoaLigature0 ,"features"])\

\f0\fs24 \CocoaLigature1 data
\f1\fs22 \CocoaLigature0 $sharefeature = sapply(strsplit(as.character(
\f0\fs24 \CocoaLigature1 data 
\f1\fs22 \CocoaLigature0 $features), " and ", fixed=TRUE), "[", 1)\

\f0\fs24 \CocoaLigature1 data[!grepl("hr", data$sharefeature)
\f1\fs22 \CocoaLigature0 ,"sharefeature"] = "no_share_features"\

\f0\fs24 \CocoaLigature1 \
#d = data[data$sharefeature=="1hr of clicks", ]\
\
summary = ddply(data, .(features), summarize, total_clicks=sum(real_value), total_estimated_clicks=sum(predicted_value), total_urls=length(real_value), underestimated_clicks =sum(diff[diff<0]), overestimated_clicks=sum(diff[diff>0]), exact_urls=length(diff[diff==0]), underestimated_urls=length(diff[diff<0]), overestimated_urls=length(diff[diff>0]))\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural

\f1\fs22 \cf0 \CocoaLigature0 summary$time = sapply(strsplit(as.character(
\f0\fs24 \CocoaLigature1 summary 
\f1\fs22 \CocoaLigature0 $features), " ", fixed=TRUE), "[", 1)\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 \CocoaLigature1 summary
\f1\fs22 \CocoaLigature0 [!grepl("hr", 
\f0\fs24 \CocoaLigature1 summary 
\f1\fs22 \CocoaLigature0 $time),"time"] = "-"\

\f0\fs24 \CocoaLigature1 summary
\f1\fs22 \CocoaLigature0 $time = factor(
\f0\fs24 \CocoaLigature1 summary 
\f1\fs22 \CocoaLigature0 $time, levels = c("1hr", "4hr", "24hr", "-"))\

\f0\fs24 \CocoaLigature1 summary 
\f1\fs22 \CocoaLigature0 $userfeature = sapply(strsplit(as.character(
\f0\fs24 \CocoaLigature1 summary
\f1\fs22 \CocoaLigature0 $features), " and ", fixed=TRUE), "[", 2)\

\f0\fs24 \CocoaLigature1 summary[
\f1\fs22 \CocoaLigature0 is.na(
\f0\fs24 \CocoaLigature1 summary
\f1\fs22 \CocoaLigature0 $userfeature),"userfeature"] = "no_user_features"\

\f0\fs24 \CocoaLigature1 summary[!grepl("hr", summary$features)
\f1\fs22 \CocoaLigature0 ,"userfeature"] = as.character(
\f0\fs24 \CocoaLigature1 summary[!grepl("hr", summary$features)
\f1\fs22 \CocoaLigature0 ,"features"])\

\f0\fs24 \CocoaLigature1 summary
\f1\fs22 \CocoaLigature0 $sharefeature = sapply(strsplit(as.character(
\f0\fs24 \CocoaLigature1 summary 
\f1\fs22 \CocoaLigature0 $features), " and ", fixed=TRUE), "[", 1)\

\f0\fs24 \CocoaLigature1 summary[!grepl("hr", summary$sharefeature)
\f1\fs22 \CocoaLigature0 ,"sharefeature"] = "no_share_features"\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural
\cf0 \
#options(digits=2)\
summary$value = -summary$underestimated_clicks/summary$total_clicks\
write.table(format(cast(summary, sharefeature ~userfeature, mean),digits=2), file=paste(prefix, "
\f0\fs24 \CocoaLigature1 _underestimate_all.txt
\f1\fs22 \CocoaLigature0 ",sep=""), sep=",", quote=FALSE, row.names=FALSE)\
summary$value = summary$overestimated_clicks/summary$total_clicks\
write.table(format(cast(summary, sharefeature ~userfeature, mean),digits=2), file=paste(prefix, "
\f0\fs24 \CocoaLigature1 _overestimate_all.txt
\f1\fs22 \CocoaLigature0 ",sep=""), sep=",", quote=FALSE, row.names=FALSE)\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 \CocoaLigature1 \
p = ggplot(summary)\
p = p + geom_bar(aes(x=sharefeature, y = underestimated_clicks/total_clicks, fill=userfeature), stat="identity",
\f2 position="dodge"
\f0 )\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural

\f1\fs22 \cf0 \CocoaLigature0 p = p + theme(axis.text.x = element_text(angle = 45, hjust = 1))\
ggsave(file=paste(prefix, "
\f0\fs24 \CocoaLigature1 _underestimate_all.pdf
\f1\fs22 \CocoaLigature0 ",sep=""), plot=p, width=10, height=6)\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 \CocoaLigature1 p = ggplot(summary)\
p = p + geom_bar(aes(x=sharefeature, y = overestimated_clicks/total_clicks, fill=userfeature), stat="identity",
\f2 position="dodge"
\f0 )\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural

\f1\fs22 \cf0 \CocoaLigature0 p = p + theme(axis.text.x = element_text(angle = 45, hjust = 1))\
ggsave(file=paste(prefix, "
\f0\fs24 \CocoaLigature1 _overestimate_all.pdf
\f1\fs22 \CocoaLigature0 ",sep=""), plot=p, width=10, height=6)\
\
summary$userfeature = sapply(strsplit(as.character(summary$userfeature), "_x_", fixed=TRUE), "[", 1)\
\
plot_heatmap_tables = function(summary, filename)\{\
b = cast(summary, sharefeature ~userfeature, mean)\
c = melt(b)\
\pard\pardeftab720

\f2\fs24 \cf0 \CocoaLigature1 c$sharefeature = factor(c$sharefeature, levels=c("4hr of clicks+impressions", "4hr of clicks", "1hr of clicks+impressions", "1hr of clicks", "4hr of shares+impressions", "4hr of impressions", "4hr of shares", "1hr of shares+impressions", "1hr of impressions", "1hr of shares","no_share_features"))\
c$userfeature = factor(c$userfeature, levels=c("no_user_features", "num_urls",  "score_median", "score_95", "score_median+score_95", "verified", "num_primary_urls", "num_followers", "all", "all_except_scores"))\
print(unique(c$sharefeature))\
p = ggplot(c, aes(userfeature, sharefeature)) \
p = p + geom_tile(aes(fill = value)) + geom_text(aes(fill = value, label = round(value, 3)), size=3)\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0 \cf0 p = p + scale_fill_gradientn(colours = rainbow(5))\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural

\f1\fs22 \cf0 \CocoaLigature0 p = p + theme(axis.text.x = element_text(angle = 45, hjust = 1))\
p = p + xlab("User-based features")+ ylab("Share-based features")+labs(fill="number of samples")\
ggsave(file=filename, plot=p, width=8, height=4)\
\}\
summary$value = -summary$underestimated_clicks/summary $total_clicks\
plot_heatmap_tables(summary, paste(prefix, "
\f0\fs24 \CocoaLigature1 _underestimate_table.pdf
\f1\fs22 \CocoaLigature0 ",sep=""))\
summary$value = summary$overestimated_clicks/summary$total_clicks\
plot_heatmap_tables(summary, paste(prefix, "
\f0\fs24 \CocoaLigature1 _overestimate_table.pdf
\f1\fs22 \CocoaLigature0 ",sep=""))\
summary$value = summary$overestimated_clicks/summary$total_estimated_clicks\
plot_heatmap_tables(summary, paste(prefix, "
\f0\fs24 \CocoaLigature1 _overestimate_bytotalestimated_table.pdf
\f1\fs22 \CocoaLigature0 ",sep=""))\
\
\
\
plot_forsharefeature = function(sharefeature)\
\{\
### under/over estimate bar graphs\
print(sharefeature)\
s = summary[summary$sharefeature==sharefeature, ]\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 \CocoaLigature1 p = ggplot(s)\
p = p + geom_bar(aes(x=sharefeature, y = underestimated_clicks/total_clicks, fill=userfeature), stat="identity",
\f2 position="dodge"
\f0 )\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural

\f1\fs22 \cf0 \CocoaLigature0 p = p + theme(axis.text.x = element_text(angle = 45, hjust = 1))\
ggsave(file=paste(prefix, "
\f0\fs24 \CocoaLigature1 _underestimate_",sharefeature,".pdf
\f1\fs22 \CocoaLigature0 ",sep=""), plot=p, width=10, height=6)\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 \CocoaLigature1 p = ggplot(s)\
p = p + geom_bar(aes(x=sharefeature, y = overestimated_clicks/total_clicks, fill=userfeature), stat="identity",
\f2 position="dodge"
\f0 )\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural

\f1\fs22 \cf0 \CocoaLigature0 p = p + theme(axis.text.x = element_text(angle = 45, hjust = 1))\
ggsave(file=paste(prefix, "
\f0\fs24 \CocoaLigature1 _overestimate_",sharefeature,".pdf
\f1\fs22 \CocoaLigature0 ",sep=""), plot=p, width=10, height=6)\
\
\
### diff histograms\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 \CocoaLigature1 d = data[data$sharefeature==sharefeature,]\
p = ggplot(data=d)\
if (grepl("LogClicks", prefix)) \{\
  p = p + geom_histogram(aes(x=diff), binwidth=.5) + facet_wrap(~ userfeature)\
\} else \{\
  p = p + geom_histogram(aes(x=diff), binwidth=100) + facet_wrap(~ userfeature)\
\}\
ggsave(file=paste(prefix,"_diffestimate_histogram_", sharefeature,".pdf",sep=""), plot=p, 
\f1\fs22 \CocoaLigature0 width=10, height=6
\f0\fs24 \CocoaLigature1 )\
if (!grepl("LogClicks", prefix)) \{\
  p = p + xlim(c(-1000,1000))\
  ggsave(file=paste(prefix,"_diffestimate_histogram_cutoff_", sharefeature,".pdf",sep=""), plot=p, 
\f1\fs22 \CocoaLigature0 width=10, height=6
\f0\fs24 \CocoaLigature1 )\
\}\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural

\f1\fs22 \cf0 \CocoaLigature0 d2 = d[grepl("\\\\+", d$userfeature) | grepl("all", d$userfeature) | grepl("no", d$userfeature),]\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 \CocoaLigature1 p = ggplot(data=d2)\
p = p + geom_freqpoly(aes(x=diff, colour= userfeature), binwidth=1)\
  p = p + xlim(c(-20,20)) \
ggsave(file=paste(prefix,"_diffestimate_histogram_someuserfeatures", sharefeature,".pdf",sep=""), plot=p, 
\f1\fs22 \CocoaLigature0 width=10, height=6
\f0\fs24 \CocoaLigature1 )\
\
\
line_data = 
\f2 data.frame(x = c(0.1,1,2,3,4,5), y = c(0.1,5,55,550,3000,5000))
\f0 \
\
if (!grepl("LogClicks", prefix))\
  \{\
	p = ggplot(data=d)\
	p =  p + geom_boxplot(aes(x=factor(predicted_value), y=real_value, colour=userfeature))\
	p = p + scale_y_log10()\
	ggsave(file=paste(prefix, "_diffestimate_boxplot_", sharefeature,".pdf",sep=""), plot=p, 
\f1\fs22 \CocoaLigature0 width=10, height=6
\f0\fs24 \CocoaLigature1 )\
\
	width = 250\
	d2$bin = round(d2$real_value/width)*width\
	a = ddply(d2, .(predicted_value, bin,userfeature), summarize, count=length(features))\
	p = ggplot()\
	p =  p + geom_tile(data=a, aes(x=factor(predicted_value), y=bin, fill=log(count+1))) + facet_wrap(~userfeature)\
	p = p + 
\f2 geom_text(data=a, aes(
\f0 x=factor(predicted_value), y= bin,
\f2 fill = log(count+1), label = count), size=3)
\f0 \
\pard\pardeftab720

\f2 \cf0 	p = p + geom_line(data = line_data, aes(x = x, y = y), colour = "grey")
\f0 \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 	p = p + scale_fill_gradientn(colours = rainbow(5))\
	p = p + scale_y_sqrt(limits=c(-1,5000))\
	p = p + ylab("real value") + xlab("predicted value")\
	ggsave(file=paste(prefix,"_diffestimate_heatmap_", sharefeature,".pdf",sep=""), plot=p, 
\f1\fs22 \CocoaLigature0 width=10, height=6
\f0\fs24 \CocoaLigature1 )\
\
	d$unlog_real_value = exp(d$real_value)-1\
	d$unlog_pred_value = exp(d$predicted_value)-1\
	d[d$unlog_real_value <=0 ,"real_class"] = 0 \
	d[d$unlog_real_value>0 & d$unlog_real_value <10 ,"real_class"] = 5\
	d[d$unlog_real_value>=10 & d$unlog_real_value <100 ,"real_class"] = 55\
	d[d$unlog_real_value>=100 & d$unlog_real_value <1000 ,"real_class"] = 550\
	d[d$unlog_real_value>=1000 & d$unlog_real_value <5000 ,"real_class"] = 3000\
	d[d$unlog_real_value>=5000 ,"real_class"] = 5000\
	d[d$unlog_pred_value <=0 ,"predicted_class"] = 0 \
	d[d$unlog_pred_value>0 & d$unlog_pred_value <10 ,"predicted_class"] = 5\
	d[d$unlog_pred_value>=10 & d$unlog_pred_value <100 ,"predicted_class"] = 55\
	d[d$unlog_pred_value>=100 & d$unlog_pred_value <1000 ,"predicted_class"] = 550\
	d[d$unlog_pred_value>=1000 & d$unlog_pred_value <5000 ,"predicted_class"] = 3000\
	d[d$unlog_pred_value>=5000 ,"predicted_class"] = 5000\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural

\f1\fs22 \cf0 \CocoaLigature0 \
\
 	 d$userfeature = sapply(strsplit(as.character(
\f0\fs24 \CocoaLigature1 d
\f1\fs22 \CocoaLigature0 $userfeature), "_x_", fixed=TRUE), "[", 1)\
	 d = d[grepl("\\\\+", d$userfeature) | d$userfeature=="all" | grepl("no", d$userfeature),]\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 \CocoaLigature1 	width = 250\
	a = ddply(d, .(predicted_class, real_class,userfeature), summarize, count=length(features))\
	for (i in all_classes) \{\
	  for (j in all_classes) \{\
	    for (uf in unique(a$userfeature)) \{\
		if (dim(a[a$predicted_class==i & a$real_class==j & a$userfeature==uf,])[1] == 0) \{\
		  a <- rbind(a, data.frame(predicted_class=i, real_class=j, count=0, userfeature=uf))\
		\}\
	     \} \
	  \}\
	\}\
	total_rows = ddply(a, .(real_class,userfeature), summarize, total=sum(count))\
	a = merge(a, total_rows, by=c("real_class", "userfeature"))\
	a$count_colour = a$count/a$total\
	a$count_value = paste(round(a$count/a$total,2)*100, "%")\
	a[a$predicted_class != a$real_class, "count_value"] = ""\
	p = ggplot(data=a)\
	p =  p + geom_tile(aes(x=factor(predicted_class), y= factor(real_class), fill=count_colour)) + facet_wrap(~userfeature)\
	p = p + 
\f2 geom_text(aes(
\f0 x=factor(predicted_class), y= factor(real_class), 
\f2 fill = 
\f0 count_colour
\f2 , label = count_value), size=3)
\f0 \
\pard\pardeftab720

\f2 \cf0 	#p = p + geom_line(data = line_data, aes(x = x, y = y), colour = "grey")
\f0 \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 	p = p + scale_fill_gradientn(colours = matlab.like(7),
\f2 limits = c(0,1)
\f0 )\
	p = p  + ylab("real value") + xlab("predicted value") + labs(fill='number of samples')\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural

\f1\fs22 \cf0 \CocoaLigature0 	ggsave(file=paste(prefix, "
\f0\fs24 \CocoaLigature1 _confusion_matrices_",sharefeature,".pdf
\f1\fs22 \CocoaLigature0 ",sep=""), plot=p, width=11, height=4)\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 \
  \} else \{ # linear regression\
	print("!!!")\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 \CocoaLigature1 	width = 2\
	d$bin = round(d$real_value/width)*width\
	d$bin_pred = round(d$predicted_value/width)*width\
	p = ggplot(data=d)\
	p =  p + geom_boxplot(aes(x=factor(bin_pred), y=real_value, colour=userfeature))\
	p = p + scale_y_log10()\
	p = p + ylab("log real value") + xlab("log predicted value")\
	ggsave(file=paste(prefix, "_diffestimate_boxplot_", sharefeature,".pdf",sep=""), plot=p, 
\f1\fs22 \CocoaLigature0 width=10, height=6
\f0\fs24 \CocoaLigature1 )\
\
	d$unlog_real_value = exp(d$real_value)-1\
	d$unlog_pred_value = exp(d$predicted_value)-1\
	d[d$unlog_pred_value <=0 ,"predicted_class"] = 0 \
	d[d$unlog_pred_value>0 & d$unlog_pred_value <10 ,"predicted_class"] = 5\
	d[d$unlog_pred_value>=10 & d$unlog_pred_value <100 ,"predicted_class"] = 55\
	d[d$unlog_pred_value>=100 & d$unlog_pred_value <1000 ,"predicted_class"] = 550\
	d[d$unlog_pred_value>=1000 & d$unlog_pred_value <5000 ,"predicted_class"] = 3000\
	d[d$unlog_pred_value>=5000 ,"predicted_class"] = 5000\
	d[d$unlog_real_value <=0 ,"real_class"] = 0 \
	d[d$unlog_real_value>0 & d$unlog_real_value <10 ,"real_class"] = 5\
	d[d$unlog_real_value>=10 & d$unlog_real_value <100 ,"real_class"] = 55\
	d[d$unlog_real_value>=100 & d$unlog_real_value <1000 ,"real_class"] = 550\
	d[d$unlog_real_value>=1000 & d$unlog_real_value <5000 ,"real_class"] = 3000\
	d[d$unlog_real_value>=5000 ,"real_class"] = 5000\
	all_classes = c(0,5,55,550,3000,5000)\
\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural

\f1\fs22 \cf0 \CocoaLigature0  	 d$userfeature = sapply(strsplit(as.character(
\f0\fs24 \CocoaLigature1 d
\f1\fs22 \CocoaLigature0 $userfeature), "_x_", fixed=TRUE), "[", 1)\
	 d = d[grepl("\\\\+", d$userfeature) | d$userfeature=="all" | grepl("no", d$userfeature),]\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 \CocoaLigature1 	width = 250\
	a = ddply(d, .(predicted_class, real_class,userfeature), summarize, count=length(features))\
	for (i in all_classes) \{\
	  for (j in all_classes) \{\
	    for (uf in unique(a$userfeature)) \{\
		if (dim(a[a$predicted_class==i & a$real_class==j & a$userfeature==uf,])[1] == 0) \{\
		  a <- rbind(a, data.frame(predicted_class=i, real_class=j, count=0, userfeature=uf))\
		\}\
	     \} \
	  \}\
	\}\
	total_rows = ddply(a, .(real_class,userfeature), summarize, total=sum(count))\
	a = merge(a, total_rows, by=c("real_class", "userfeature"))\
	a$count_colour = a$count/a$total\
	a$count_value = paste(round(a$count/a$total,2)*100, "%")\
	a[a$predicted_class != a$real_class, "count_value"] = ""\
	p = ggplot(data=a)\
	p =  p + geom_tile(aes(x=factor(predicted_class), y= factor(real_class), fill=count_colour)) + facet_wrap(~userfeature)\
	p = p + 
\f2 geom_text(aes(
\f0 x=factor(predicted_class), y= factor(real_class), 
\f2 fill = 
\f0 count_colour
\f2 , label = count_value), size=3)
\f0 \
\pard\pardeftab720

\f2 \cf0 	#p = p + geom_line(data = line_data, aes(x = x, y = y), colour = "grey")
\f0 \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 	p = p + scale_fill_gradientn(colours = matlab.like(7),
\f2 limits = c(0,1)
\f0 )\
	p = p  + ylab("real value") + xlab("predicted value") + labs(fill='number of samples')\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural

\f1\fs22 \cf0 \CocoaLigature0 	ggsave(file=paste(prefix, "
\f0\fs24 \CocoaLigature1 _confusion_matrices_",sharefeature,".pdf
\f1\fs22 \CocoaLigature0 ",sep=""), plot=p, width=11, height=4)\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 \CocoaLigature1 \
\
	width = 250 # this is now the same as with the other method\
	d$bin_real_value = round(d$unlog_real_value/width)*width \
	a = ddply(d, .(predicted_class, bin_real_value,userfeature), summarize, count=length(features))\
	p = ggplot(data=a)\
	p =  p + geom_tile(aes(x=factor(predicted_class), y= bin_real_value, fill=log(count+1))) + facet_wrap(~userfeature)\
\pard\pardeftab720

\f2 \cf0 	p = p + geom_line(data = line_data, aes(x = x, y = y), colour = "grey")
\f0 \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 	p = p + scale_fill_gradientn(colours = rainbow(5))\
	p = p + 
\f2 geom_text(aes(
\f0 x=factor(predicted_class), y= bin_real_value,
\f2 fill = log(count+1), label = count), size=3)
\f0 \
	p = p + scale_y_sqrt(limits=c(-1,5000))\
	p = p  + ylab("real value") + xlab("predicted value")\
	ggsave(file=paste(prefix,"_diffestimate_heatmap_", sharefeature,".pdf",sep=""), plot=p, 
\f1\fs22 \CocoaLigature0 width=10, height=8
\f0\fs24 \CocoaLigature1 )\
\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f1\fs22 \cf0 \CocoaLigature0 \
  \}\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural
\cf0 \}\
\
for (i in levels(factor(data$sharefeature)))\
\{\
print(i)\
plot_forsharefeature(i)\
\}\
\
\
\
\
}