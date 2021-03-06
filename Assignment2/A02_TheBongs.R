chooseCRANmirror(graphics=FALSE, ind=1)
knitr::opts_chunk$set(echo = TRUE)
#Q1
#install.packages("plotly")
library(plotly)
library(MASS)
require(data.table)
library(dplyr)

#Q4
#install.packages("ggmap")
library(ggmap)
library(ggplot2)
library(shiny)

#Q5
#install.packages("RColorBrewer")
#install.packages("tidyr")
#install.packages("textdata")
#install.packages("wordcloud")
library(dplyr)
library(tidytext)
library(wordcloud)
library(wordcloud)
library(reshape2)

####################################################################################################################################################################################



#QUESTION 1
data=fread("fitness_data.csv")
d=na.omit(data)
sub_IDs=sort(unique(d$subID))       #obtaining all the ids
#plotting the graph
graph <- plot_ly(d, x = ~d$`timestamp (in seconds)`,
                 transforms = list( 
                   list(type = 'filter',
                        target = d$subID,
                        operation = '=',
                        value = sub_IDs[1]
                   ))) %>% 
  add_trace(y = ~d$`Body Temperature (Celsius)`, name = "Body Temperatue",mode='lines+markers')  %>%
  add_trace(y = ~d$`heartrate during activity (bpm)`, name = "heartrate during activity (bpm)",mode='lines+markers', visible = F)  %>%
  add_trace(y = ~d$`Acceleration (x axis) in m/s2`, name = "Acceleration (x axis) in m/s2",mode='lines+markers', visible = F)  %>%
  add_trace(y = ~d$`Acceleration (y axis) in m/s2`, name = "Acceleration (y axis) in m/s2",mode='lines+markers', visible = F)  %>%
  add_trace(y = ~d$`Acceleration (z axis) in m/s2`, name = "Acceleration (z axis) in m/s2",mode='lines+markers', visible = F) %>%
  add_trace(y = ~d$`Gyroscope (x axis) in rad/s`, name = "Gyroscope (x axis) in rad/s",mode='lines+markers',visible = F) %>%
  add_trace(y = ~d$`Gyroscope (y axis) in rad/s`, name = "Gyroscope (y axis) in rad/s",mode='lines+markers', visible = F) %>%
  add_trace(y = ~d$`Gyroscope (z axis) in rad/s`, name = "Gyroscope (z axis) in rad/s", mode='lines+markers',visible = F) %>%
  add_trace(y = ~d[,12], name = "Magnetometer (x axis) in Î¼T",mode='lines+markers',mode='lines+markers', visible = F) %>%
  add_trace(y = ~d[,13], name = "Magnetometer (y axis) in Î¼T",mode='lines+markers',mode='lines+markers', visible = F) %>%
  add_trace(y = ~d[,14], name = "Magnetometer (z axis) in Î¼T",mode='lines+markers', mode='lines+markers',visible = F) %>%
  layout(
    title = "Fitness data",
    
    xaxis = list(
      title="x(timestamp in seconds)",
      rangeselector = list(buttons =
                             list(list(
                               count = 3, 
                               stepmode = "backward"),
                               list(step = "all"))),
      rangeslider = list(type = "time")),
    yaxis = list(title = "y"),
    updatemenus = list(
      list(
        type='dropdown',
        active=0,
        x=1.3,y = 0.9,
        buttons= list(
          list(method="restyle",args=list("transforms[0].value",sub_IDs[1]),
               label=sub_IDs[1]),
          list(method="restyle",args=list("transforms[0].value",sub_IDs[2]),
               label=sub_IDs[2]),
          list(method="restyle",args=list("transforms[0].value",sub_IDs[3]),
               label=sub_IDs[3]),
          list(method="restyle",args=list("transforms[0].value",sub_IDs[4]),
               label=sub_IDs[4]),
          list(method="restyle",args=list("transforms[0].value",sub_IDs[5]),
               label=sub_IDs[5]),
          list(method="restyle",args=list("transforms[0].value",sub_IDs[6]),
               label=sub_IDs[6]),
          list(method="restyle",args=list("transforms[0].value",sub_IDs[7]),
               label=sub_IDs[7]),
          list(method="restyle",args=list("transforms[0].value",sub_IDs[8]),
               label=sub_IDs[8]),
          list(method="restyle",args=list("transforms[0].value",sub_IDs[9]),
               label=sub_IDs[9])
        )),
      list(
        x=1.5,y = 0.75,
        buttons = list(
          list(method = "restyle",
               args = list("visible", list(TRUE, FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE)),
               label = "Body Temperature (Celsius)"),
          
          list(method = "restyle",
               args = list("visible",list(FALSE, TRUE ,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE)),
               label = "heartrate during activity (bpm)" ),
          
          list(method = "restyle",
               args = list("visible",list(FALSE,FALSE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE)),
               label = "Acceleration (x axis) in m/s2"  ),
          
          list(method = "restyle",
               args = list("visible",list(FALSE,FALSE,FALSE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE)),
               label = "Acceleration (y axis) in m/s2"),
          
          list(method = "restyle",
               args = list("visible",list(FALSE,FALSE,FALSE,FALSE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE)),
               label = "Acceleration (z axis) in m/s2"),
          
          list(method = "restyle",
               args = list("visible",list(FALSE,FALSE,FALSE,FALSE,FALSE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE)),
               label = "Gyroscope (x axis) in rad/s"),
          list(method = "restyle",
               args = list("visible",list(FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,TRUE,FALSE,FALSE,FALSE,FALSE)),
               label = "Gyroscope (y axis) in rad/s"),
          list(method = "restyle",
               args = list("visible",list(FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,TRUE,FALSE,FALSE,FALSE)),
               label = "Gyroscope (z axis) in rad/s"),
          
          list(method = "restyle",
               args = list("visible",list(FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,TRUE,FALSE,FALSE)),
               label = "Magnetometer (x axis) in Î¼T"),
          list(method = "restyle",
               args = list("visible",list(FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,TRUE,FALSE)),
               label = "Magnetometer (y axis) in Î¼T"),
          list(method = "restyle",
               args = list("visible",list(FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,TRUE)),
               label = "Magnetometer (z axis) in Î¼T")
        ))))

print(graph)


####################################################################################################################################################################################
# QUESTION 2

#Source: https://zyxo.wordpress.com/2008/12/30/oversampling-or-undersampling/

subject_df  = read.csv("subject.csv") #reading the csv file

print(subject_df)

#Checking for undersampling or oversampling in the csv
#We find that the current data is highly imbalanced. 

subject_aggregate_df = aggregate(subject_df$Dominant.hand, by=list(Sex=subject_df$Sex), FUN=length)
barplot(subject_aggregate_df$x,names.arg = subject_aggregate_df$Sex, ylab = "Population")

#The remedy to the problem is either to oversample the Female population or under sample the males
#Oversampling the females seems to be a better approach


####################################################################################################################################################################################
#QUESTION 3

fitness_df = read.csv("fitness_data.csv")
activities_df = read.csv("activities.csv")

#Through these datasets, we notice that there exists only one non-medical field recorded, which is activity ID.
#There are a total of 17 activities being performed, and hence Cluster sampling technique can be utilised to sample the subjects.

#Answer 3:
#Through these datasets, we notice that there exists only one non-medical field recorded, which is activity ID.
#There are a total of 17 activities being performed, and hence Cluster sampling technique can be utilised to sample the subjects.





####################################################################################################################################################################################
#QUESTION 4

#INPUTTING GOOGLE API KEY #INTERNET CONNECTION REQUIRED TO RUN THIS
register_google(key = 'AIzaSyDwCTPxbxMRd-nYr9b5zNs2FW8jbGluJe0')
map <- get_map(location = 'India', zoom = 5)
#let's read in the csv
election_df = read.csv("Lok Sabha-2014 data.csv")
#print(election_df)

#Q4 SHINY APP
#We are visualising the spread of voters for each party as it tells us how much prevelance or success a party has had in a certain regioon.
#The circles on map locate the region and the radius of each circle is the corresponding margin of votes by which they won.
#We are using a shiny party selector drop down menu. 

ui<-shinyUI(fluidPage(
  titlePanel("Election Results"),
  # Your input selection
  sidebarPanel(
    selectInput("party", "Choose you Input:", choices = unique(election_df$PARTY))
  ),
  # Show the selected plot
  mainPanel(
    plotOutput("whichplot")
  )
))

server<-shinyServer(function(input, output) {
  
  # Fill in the spot we created for a plot
  output$whichplot <- renderPlot({
    
    bjp = subset(election_df,election_df$PARTY ==input$party,)
    points <- ggmap(map) + geom_point(aes(x = longitude, y = latitude,size = MARGIN), data = bjp ,alpha = .4,color= "orange")
    points <- points + scale_size_area(name = "MARGIN WIN")
    points
  })
})
shinyApp(ui, server)

#IMPORTANT!
#NOTE AS SHINY APPLICATIONS ARE NOT SUPPORTED IN STATIC HTML PAGES
#THE ABOVE APP HAS BEEN IMPLEMENTED BELOW BUT WITHOUT THE ABILITY TO CHOSE A PARTY
#PLOTTING BY PARTY
#YOU CAN PUNCH IN ANY PARTY NAME AND IT WILL SHOW A MAP DEPICTING THE WIN REGION AND MARGIN OF WIN IS DEPICTED BY THE RADIUS
bjp = subset(election_df,election_df$PARTY =="Bharatiya Janata Party",)
cong = subset(election_df,election_df$PARTY =="Indian National Congress",)
points <- ggmap(map) + geom_point(aes(x = longitude, y = latitude,size = MARGIN), data = bjp ,alpha = .4,color= "orange") 
points <- points + scale_size_area(name = "MARGIN WIN")
points


#WE NEED TO PLOT A BAR GRAPH DENOTING THE NUMBER OF CONSTITUENCIES WON BY EVERY PARTY
party_aggregate = aggregate(election_df$PARTY, by=list(PARTY=election_df$PARTY), FUN=length)
party_aggregate = party_aggregate[order(-(party_aggregate$x)),] #- helps us sort in descending order
#print(party_aggregate)
par(mar=c(12,6.5,2.5,0.5),mgp=c(5,1,0)) 
barplot(party_aggregate$x,names.arg = party_aggregate$PARTY,las=2,col="ORANGE",main="Election Results",ylab = "Constituencies")
#xlab  missing as the party name overlaps it.




####################################################################################################################################################################################
#QUESTION 5

#reading tweets
filePath <- "tweets.txt"
text <- readLines(filePath)
text <- c(text)

text_df <- tibble(line = 1:12596, text = text)

text_df<-text_df %>%
  unnest_tokens(word, text)

text_df%>%
  count(word, sort = TRUE)

nrc_joy <- get_sentiments("nrc") %>%
  filter(sentiment == "joy")

text_df %>%
  inner_join(nrc_joy) %>%
  count(word, sort = TRUE)

#showing the cloud

par(mar=c(0,1.5,0.5,0.5),mgp=c(10,1,0)) #do not change par

#showing the cloud of positive and negative values
cloud = text_df %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("red", "blue"), #red means negative
                   max.words = 100)

cloud






