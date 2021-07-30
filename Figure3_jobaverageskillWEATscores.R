filename = "OUTPUT_ESCO_EN"
output_filename = paste(filename,"_avg.csv", sep='')

df  = read.csv(paste(filename,".csv",sep=''),sep=";" )
codes = unique(df$Code)
skillWEATscores = matrix(0,length(codes),1)
MaleEmploymentRatio = matrix(0,length(codes),1)

for (i in 1:length(codes)){ subset = df[df$Code==codes[i],]
 skillWEATscores[i] = mean(subset$WEATscore)
 MaleEmploymentRatio[i] = mean(subset$MalePercentage)
}

OUT = cbind(MaleEmploymentRatio,skillWEATscores)
cor(MaleEmploymentRatio,skillWEATscores)

write.csv(OUT,file=output_filename)

