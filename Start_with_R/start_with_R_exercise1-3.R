rooms <- c(1, 2, 1, 3, 1, NA, 3, 1, 3, 2, 1, NA, 1, 8, 3, 1, 4, NA, 1, 3, 1, 2, 1, 7, 1, NA)

#Vector greater than 2
rooms_no_na <- na.omit(rooms)
rooms_above_2 <- rooms_no_na[rooms_no_na > 2]
length(rooms_above_2)

#Median
median(rooms, na.rm = TRUE)











