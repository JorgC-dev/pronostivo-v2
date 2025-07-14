SELECT
    CAST(f.OrderDate AS DATE) AS [sales_date],
    CAST(SUM(f.SalesAmount) AS INT) AS TotalVentas
FROM
    FactInternetSales AS f
WHERE 
    f.OrderDate < '2013-01-01'
GROUP BY
    CAST(f.OrderDate AS DATE)
ORDER BY
    CAST(f.OrderDate AS DATE) ASC;