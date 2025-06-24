SELECT
    CAST(OrderDate AS DATE) AS Date
    ,YEAR([OrderDate]) AS [YEAR]
    ,MONTH([OrderDate]) AS [MONTH]
    ,DAY([OrderDate]) AS [Day]
    ,DATEPART(weekday, [OrderDate]) - 1 AS [DayOfWeek]
    ,CASE
        WHEN DATEPART(weekday, [OrderDate]) IN (7, 1) THEN 1
        ELSE 0
    END AS IsWeekend
    ,[ProductKey]
    ,Sum([OrderQuantity]) AS OrderQuantity
    ,Sum([UnitPrice]) AS UnitPrice
    ,Sum([UnitPriceDiscountPct]) As UnitPriceDiscountPct
    ,Sum([DiscountAmount]) AS DiscountAmount
    ,Sum([ProductStandardCost]) AS ProductStandardCost
    ,Sum([SalesAmount]) AS SalesAmount
    FROM [AdventureWorksDW2022].[dbo].[FactInternetSales]
    WHERE
    [FactInternetSales].OrderDate < '2013-12-01'
    group by YEAR([OrderDate]),MONTH([OrderDate]),DAY([OrderDate]),DATENAME(weekday,[OrderDate]),[OrderDate],[ProductKey]
    order by YEAR([OrderDate]) ASC,MONTH([OrderDate]) ASC,DAY([OrderDate]) ASC