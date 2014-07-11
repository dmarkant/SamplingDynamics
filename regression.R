setwd("~/code/SamplingDynamics/")

library(lme4)
library(plyr)
library(nnet)

library(languageR)

library(lmerTest)

data = as.data.frame(read.csv("dfe_by_game.csv"))
data$group = factor(data$group)
data$domain = factor(data$domain)
data$switch_grp = factor(data$switch_grp)
data$switchfreq = as.numeric(as.character(data$switchfreq))
data$gamble_lab = factor(data$gamble_lab)
data$partid = factor(data$partid)
data$L_id = factor(data$L_id)
data$H_id = factor(data$H_id)

# Sample size
m1 = lmerTest::lmer(samplesize ~ 
                      group + 
                      session + 
                      domain +
                      pairtype +
                      ev_diff +
                      switch_grp +
                      (1|partid), data=data)



# Number of switches
m = lmerTest::lmer(switchcount ~
                     group +
                     session +
                     domain +
                     pairtype +
                     ev_diff +
                     (samplesize|partid), data=data)


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
data$sample_var = as.numeric(as.character(data$sample_var))
data$switch_or_stop = data$switched==1 | data$stopped==1
data$abs_dev = abs(data$deviation)
data$dec = as.numeric(data$switched)-1 + 2*(as.numeric(data$stopped))
data$mn_diff = as.numeric(as.character(data$mn_diff))
data$lpv = log(as.numeric(as.character(data$lpv)))

freq_data = data[data$switch_grp == 'freq',]
rare_data = data[data$switch_grp == 'rare',]



rcorr.cens(rare_data$abs_dev, rare_data$switch_or_stop, outx=TRUE)




subj_rare = unique(data[data$switch_grp=='rare',]$partid)
subj_freq = unique(data[data$switch_grp=='freq',]$partid)

cols = c("intercept", "streak_length", "sample_mean", "deviation", "abs_dev", "sample_var", "lpv", "abs_mn_diff")
cols2 = c("intercept", "sample_mean", "deviation", "abs_dev", "sample_var", "lpv", "abs_mn_diff")

rare_mat = matrix(nrow = 0, ncol=length(cols))
rare_mat2 = matrix(nrow = 0, ncol=length(cols2))

m = glm(switch_or_stop ~ 1 + streak_length + sample_mean + deviation + abs_dev + sample_var + lpv + abs(mn_diff), data=sdata, family=binomial)

for (sid in subj_rare) {
  sdata = data[data$partid==sid,]  
  
  m = glm(switch_or_stop ~ 1 + streak_length + sample_mean + deviation + abs_dev + sample_var + lpv + abs(mn_diff), data=sdata, family=binomial)
  rare_mat = rbind(rare_mat, summary(m)$coefficients[,3])
  
  dec_data = sdata[sdata$dec>0,]
  dec_data$dec = dec_data$dec - 1
  m = glm(dec ~ 1 + sample_mean + deviation + abs_dev + sample_var + lpv + abs(mn_diff), data=dec_data, family=binomial)
  
  rare_mat2 = rbind(rare_mat2, summary(m)$coefficients[,3])
}


freq_mat = matrix(nrow = 0, ncol=length(cols))
freq_mat2 = matrix(nrow = 0, ncol=length(cols2))

for (sid in subj_freq) {
  print(sid)
  sdata = data[data$partid==sid,]  
  
  m = glm(switch_or_stop ~ 1 + streak_length + sample_mean + deviation + abs_dev + lpv + abs(mn_diff), data=sdata, family=binomial)
  freq_mat = rbind(freq_mat, summary(m)$coefficients[,3])
  
  dec_data = sdata[sdata$dec>0,]
  dec_data$dec = dec_data$dec - 1
  m = glm(dec ~ 1 + sample_mean + deviation + abs_dev + lpv + abs(mn_diff), data=dec_data, family=binomial)
  
  freq_mat2 = rbind(freq_mat2, summary(m)$coefficients[,3])
}


# STAY v LEAVE
rare_df = as.data.frame(rare_mat)
#freq_df = as.data.frame(freq_mat)
names(rare_df) = cols
#names(freq_df) = cols

par(mfrow=c(3,2))
plot(density(rare_df$sample_mean), main='Sample mean', col='red', xlim=c(-10, 10))
plot(density(rare_df$deviation), main='Deviation from sample mean', col='red', ylim=c(0, .5), xlim=c(-10, 10))
plot(density(rare_df$abs_dev), main='Abs(deviation)', col='red', ylim=c(0, .4), xlim=c(-10, 10))
plot(density(rare_df$sample_var), main='Sample var', col='red', ylim=c(0, .4), xlim=c(-10, 10))
plot(density(rare_df$lpv), main='log pooled variance', col='red', ylim=c(0, .3), xlim=c(-10, 10))
plot(density(rare_df$abs_mn_diff), main='Abs(mean difference)', col='red', ylim=c(0, .3), xlim=c(-10, 10))


# SWITCH v STOP
rare_df2 = as.data.frame(rare_mat2)
freq_df2 = as.data.frame(freq_mat2)
names(rare_df2) = cols2
names(freq_df2) = cols2

par(mfrow=c(3,2))

plot(density(rare_df2$sample_mean), main='Sample mean', col='red', xlim=c(-10, 10))
lines(density(freq_df2$sample_mean), main='Sample mean', col='blue')

plot(density(rare_df2$deviation), main='Deviation from sample mean', col='red', ylim=c(0, .5), xlim=c(-10, 10))
lines(density(freq_df2$deviation), main='Deviation from sample mean', col='blue')

plot(density(rare_df2$abs_dev), main='Abs(deviation)', col='red', ylim=c(0, .4), xlim=c(-10, 10))
lines(density(freq_df2$abs_dev), main='Abs(deviation)', col='blue')

plot(density(rare_df2$sample_var), main='Sample var', col='red', ylim=c(0, .4), xlim=c(-10, 10))
lines(density(freq_df2$sample_var), main='Sample var', col='blue')

plot(density(rare_df2$lpv), main='log pooled variance', col='red', ylim=c(0, .3), xlim=c(-10, 10))
lines(density(freq_df2$lpv), main='log pooled variance', col='blue')

plot(density(rare_df2$abs_mn_diff), main='Abs(mean difference)', col='red', ylim=c(0, .3), xlim=c(-10, 10))
lines(density(freq_df2$abs_mn_diff), main='Abs(mean difference)', col='blue')


pairs(rare_df)
pairs(freq_df)

pairs(rare_df2)
pairs(freq_df2)



# SINGLE SUBJECT

sdata = data[data$partid==111,]

# LEAVE (SWITCH OR STOP) vs. STAY
par(mfrow=c(3,2))

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
m1 = glm(switch_or_stop ~ 1 + streak_length, data=sdata, family=binomial)
m2 = glm(switch_or_stop ~ 1 + streak_length + sample_out + deviation + abs_dev, data=sdata, family=binomial)
m3 = glm(switch_or_stop ~ 1 + lpv + mn_diff, data=sdata, family=binomial)

m = glm(switch_or_stop ~ 1 + streak_length + sample_mean + deviation + abs_dev + sample_var + lpv + abs(mn_diff), data=sdata, family=binomial)


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


# all together
m = glm(dec ~ 1 + sample_mean + deviation + abs_dev + lpv + abs(mn_diff), data=dec_data, family=binomial)




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






# SWITCH v STOP (rare group only)
dec_data = data[data$dec>0,]
dec_data$dec = dec_data$dec - 1

rare_dec_data = dec_data[dec_data$switch_grp=='rare',]
freq_dec_data = dec_data[dec_data$switch_grp=='freq',]

m = glmer(dec ~ 
            streak_length + 
            #sample_out + 
            #deviation + 
            #abs_dev + 
            #abs(mn_diff) +
            #lpv +
            (1|partid), 
          data=rare_dec_data, family=binomial)



m = glmer(dec ~ 
            #sample_out + 
            #deviation + 
            #abs_dev + 
            mn_diff +
            #abs(mn_diff) +
            #lpv +
            (1|partid), 
          data=freq_dec_data, family=binomial)



#m = glmer(switch_or_stop ~ 1 + 
#                           domain + 
#                           (streak_length|partid), 
#          data=rare_data, family=binomial, control=glmerControl(optimizer="bobyqa"))



# STAY v LEAVE (both groups)
m1 = glmer(switch_or_stop ~ 1 + (1|gamble_lab), data=sdata, family=binomial)
m1 = glmer(switch_or_stop ~ 1 + streak_length + (1|gamble_lab), data=sdata, family=binomial)


m ~ glm(stopped ~ streak_length, data=rare_)


m = glm(switch_or_stop ~ sample_out + deviation + abs(deviation) + mn_diff + abs(mn_diff) + lpv, data=rare_data, family=binomial)

m = glm(switch_or_stop ~ sample_out + deviation + abs(deviation) + mn_diff + abs(mn_diff) + lpv, data=freq_data, family=binomial)

m2 = glm(stopped ~ sample_out + abs(deviation), data=rare_data, family=binomial)


