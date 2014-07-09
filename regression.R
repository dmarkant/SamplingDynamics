setwd("~/code/SamplingDynamics/")

library(lme4)
library(plyr)
library(nnet)

# Evaluating consistency of switch frequency and sample size
data = as.data.frame(read.csv("dfe_by_game.csv"))
data$group = factor(data$group)
data$domain = factor(data$domain)
data$switch_grp = factor(data$switch_grp)


## All together
# Sample size
m = lmer(samplesize ~ group + domain + session + (1|partid), data=data)
m1 = lmer(samplesize ~ group + domain + session + (1|partid) + (1|gamble_lab), data=data)

# Number of switches
m =  lmer(switchcount ~ group + domain + session + (1|partid), data=data)
m1 = lmer(switchcount ~ group + domain + session + samplesize + (1|partid), data=data)
m2 = lmer(switchcount ~ group + domain + session + (samplesize|partid), data=data)

m3.1 = lmer(switchcount ~ group + domain + session + (1|gamble_lab), data=data)
m3 = lmer(switchcount ~ group + domain + session + (samplesize|partid) + (1|gamble_lab), data=data)

## Compare sample size using switch group
m1 = lmer(samplesize ~ group + domain + session + switch_grp + (1|partid), data=data)
m2 = lmer(samplesize ~ group + domain + session + (1|partid), data=data)


## Run separately based on switch group
freq_data = data[data$switch_grp == 'freq',]
rare_data = data[data$switch_grp == 'rare',]

# sample size
m = lmer(samplesize ~ group + domain + session + (1|partid), data=rare_data)
#m1 = lmer(samplesize ~ group + domain + session + (1|partid) + (1|gamble_lab), data=rare_data)

m = lmer(samplesize ~ group + domain + session + (1|partid), data=freq_data)
#m1 = lmer(samplesize ~ group + domain + session + (1|partid) + (1|gamble_lab), data=freq_data)


# Predicting switch and stop trials
data = as.data.frame(read.csv('reg_data.csv'))
data$domain = factor(data$domain)
data$gamble_lab = factor(data$gamble_lab)
data$switched = factor(data$switched)
data$group = factor(data$group)
data$sample_mean = as.numeric(as.character(data$sample_mean))
data$deviation = as.numeric(as.character(data$deviation))

data$switch_or_stop = data$switched==1 | data$stopped==1
data$abs_dev = abs(data$deviation)
data$dec = as.numeric(data$switched)-1 + 2*(as.numeric(data$stopped))
data$mn_diff = as.numeric(as.character(data$mn_diff))
data$lpv = log(as.numeric(as.character(data$lpv)))

freq_data = data[data$switch_grp == 'freq',]
rare_data = data[data$switch_grp == 'rare',]


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

sdata = data[data$partid==136,]


# LEAVE (SWITCH OR STOP) vs. STAY

# check out frequency of switch/stop as a function of streak length
r = ddply(sdata, c("streak_length"), function(df) mean(df$switch_or_stop))
names(r) = c("streak_length", "prop")
plot(r$streak_length, r$prop, ylim=c(0, 1))

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
plot(sdata$deviation, sdata$switch_or_stop)

m3 = glm(switch_or_stop ~ 1 + deviation, data=sdata, family=binomial)
plot(sdata$deviation, fitted(m3), ylim=c(0, 1))


# absolute deviation from sample mean
plot(sdata$abs_dev, sdata$switch_or_stop)



m4 = glm(switch_or_stop ~ 1 + abs_dev, data=sdata, family=binomial)
plot(sdata$abs_dev, fitted(m4))


# streak deviation
plot(sdata$str_deviation, sdata$switch_or_stop)

# absolute streak deviation
plot(abs(sdata$str_deviation), sdata$switch_or_stop)


# mean difference
m5 = glm(switch_or_stop ~ 1 + mn_diff, data=sdata, family=binomial)
m5 = glm(switch_or_stop ~ 1 + abs(mn_diff), data=sdata, family=binomial)


m1 = glm(switch_or_stop ~ 1 + streak_length, data=sdata, family=binomial)

m2 = glm(switch_or_stop ~ 1 + streak_length + sample_out + deviation + abs_dev, data=sdata, family=binomial)

m3 = glm(switch_or_stop ~ 1 + lpv + mn_diff, data=sdata, family=binomial)

m = glm(switch_or_stop ~ 1 + streak_length + sample_out + deviation + abs_dev, data=sdata, family=binomial)


# SWITCH vs. STOP (excluding STAY)

dec_data = sdata[sdata$dec>0,]
dec_data$dec = dec_data$dec - 1

m = glm(dec ~ 1 + mn_diff, data=dec_data, family=binomial)
plot(dec_data$mn_diff, dec_data$dec)

m1 = glm(dec ~ 1 + abs(mn_diff), data=dec_data, family=binomial)
plot(abs(dec_data$mn_diff), dec_data$dec)

m2 = glm(dec ~ 1 + abs(mn_diff) + lpv, data=dec_data, family=binomial)
plot(abs(dec_data$lpv), dec_data$dec)



m3 = glm(dec ~ 1 + sample_out + sample_mean + abs_dev, data=dec_data, family=binomial)




# JUST SWITCHES

# baseline
m0 = glm(switched ~ 1, data=sdata, family=binomial)

# streak length -- doesn't make much sense without including stop decisions as well

# most recent outcome
r = ddply(sdata, c("sample_out"), function(df) mean(as.numeric(df$switched)-1))
names(r) = c("sample_out", "prop")

m2 = glm(switched ~ 1 + sample_out, data=sdata, family=binomial)
plot(sdata$sample_out, fitted(m2), ylim=c(0, .5))
lines(r$sample_out, r$prop, col='blue')

# deviation from sample mean
m3 = glm(switched ~ 1 + deviation, data=sdata, family=binomial)
plot(sdata$deviation, fitted(m2), ylim=c(0.15, .3))
lines(sdata$deviation, as.numeric(sdata$switched)-1, col='blue')

# absolute deviation from sample mean
m4 = glm(switched ~ 1 + abs(deviation), data=sdata, family=binomial)
plot(abs(sdata$deviation), fitted(m2), ylim=c(0, .3))
lines(abs(r$deviation), r$prop, col='blue')



# streak deviation
plot(sdata$str_deviation, sdata$switched)

# absolute streak deviation
plot(abs(sdata$str_deviation), sdata$switched)
m5 = glm(switched ~ 1 + abs(str_deviation), data=sdata, family=binomial)


# MULTINOMIAL

m = multinom(dec ~ 1 + sample_out, data=sdata)

m = multinom(dec ~ 1 + abs(deviation), data=sdata)
p = predict(m, type="probs")



m1 = glmer(switch_or_stop ~ 1 + (1|gamble_lab), data=sdata, family=binomial)
m1 = glmer(switch_or_stop ~ 1 + streak_length + (1|gamble_lab), data=sdata, family=binomial)

m1 = glmer(switched ~ 1 + domain + (1|gamble_lab), data=sdata, family=binomial)
m2 = glmer(switched ~ 1 + domain + streak_length + (1|gamble_lab), data=sdata, family=binomial)
m3 = glmer(switched ~ 1 + domain + sample_mean + deviation + (1|gamble_lab), data=sdata, family=binomial)



# SWITCH AND STOP
m = glmer(switch_or_stop ~ 1 + domain + (streak_length|partid), data=rare_data, family=binomial, control=glmerControl(optimizer="bobyqa"))



# STOPPING DECISIONS ONLY (comparing rare and frequency switchers)

ind_freq = as.numeric(data$switch_grp=='freq')
ind_rare = as.numeric(data$switch_grp=='rare')



m = glm(stopped ~ sample_out + deviation, data=rare_data, family=binomial)
m2 = glm(stopped ~ sample_out + abs(deviation), data=rare_data, family=binomial)


