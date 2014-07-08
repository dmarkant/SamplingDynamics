setwd("~/code/SamplingDynamics/")

library(lme4)
library(plyr)

data = as.data.frame(read.csv('reg_data.csv'))
data$domain = factor(data$domain)
data$gamble_lab = factor(data$gamble_lab)
data$switched = factor(data$switched)
data$group = factor(data$group)
data$sample_mean = as.numeric(as.character(data$sample_mean))
data$deviation = as.numeric(as.character(data$deviation))

dims = list(unique(data$partid), c("partid", "m1_aic"))
df = data.frame(matrix(vector(), 12, 2, dimnames=dims), stringsAsFactors=F)

domainloss = vector()

for (sid in unique(data$partid)) {
  sdata = data[data$partid==sid,]  
  
  m1 = glmer(switched ~ 1 + domain + (1|gamble_lab), data=sdata, family=binomial)
  print(summary(m1))
  
  domainloss = c(domainloss, fixef(m1)['domainloss'])
  
  print(fixef(m1)['domainloss'])
  
}

sdata = data[data$partid==120,]
sdata$switch_or_stop = sdata$switched==1 | sdata$stopped==1
sdata$abs_dev = abs(sdata$deviation)

# SWITCH OR STOP

# check out frequency of switch/stop as a function of streak length
r = ddply(sdata, c("streak_length"), function(df) mean(df$switch_or_stop))
names(r) = c("streak_length", "prop")
plot(r$streak_length, r$prop, ylim=c(0, 1))
plot(log(r$streak_length), r$prop)

m1 = glm(switch_or_stop ~ 1 + streak_length, data=sdata, family=binomial)
plot(sdata$streak_length, fitted(m1), ylim=c(0, 1))
lines(r$streak_length, r$prop, col='blue')

# most recent outcome
r = ddply(sdata, c("sample_out"), function(df) mean(df$switch_or_stop))
names(r) = c("sample_out", "prop")

m2 = glm(switch_or_stop ~ 1 + sample_out, data=sdata, family=binomial)
plot(sdata$sample_out, fitted(m2), ylim=c(0, 1))
lines(r$sample_out, r$prop, col='blue')

# deviation from sample mean
r = ddply(sdata, c("deviation"), function(df) mean(df$switch_or_stop))
names(r) = c("deviation", "prop")

m3 = glm(switch_or_stop ~ 1 + deviation, data=sdata, family=binomial)
plot(sdata$deviation, fitted(m2), ylim=c(0, .3))
lines(r$deviation, r$prop, col='blue')

# absolute deviation from sample mean


# JUST SWITCH

# streak length -- doesn't make much sense without including stop decisions as well

# most recent outcome
r = ddply(sdata, c("sample_out"), function(df) mean(as.numeric(df$switched)-1))
names(r) = c("sample_out", "prop")

m2 = glm(switched ~ 1 + sample_out, data=sdata, family=binomial)
plot(sdata$sample_out, fitted(m2), ylim=c(0, .5))
lines(r$sample_out, r$prop, col='blue')

# deviation from sample mean
r = ddply(sdata, c("deviation"), function(df) mean(as.numeric(df$switched)-1))
names(r) = c("deviation", "prop")
plot(r$deviation, r$prop)

m3 = glm(switched ~ 1 + deviation, data=sdata, family=binomial)
plot(sdata$deviation, fitted(m2), ylim=c(0, .3))
lines(r$deviation, r$prop, col='blue')

# absolute deviation from sample mean










m1 = glmer(switch_or_stop ~ 1 + (1|gamble_lab), data=sdata, family=binomial)
m1 = glmer(switch_or_stop ~ 1 + streak_length + (1|gamble_lab), data=sdata, family=binomial)

m1 = glmer(switched ~ 1 + domain + (1|gamble_lab), data=sdata, family=binomial)
m2 = glmer(switched ~ 1 + domain + streak_length + (1|gamble_lab), data=sdata, family=binomial)
m3 = glmer(switched ~ 1 + domain + sample_mean + deviation + (1|gamble_lab), data=sdata, family=binomial)




sum# for each participant, run a series of logistic regressions

