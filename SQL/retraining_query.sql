SELECT
    CAST(f.OrderDate AS DATE) AS [sales_date],
    CAST(SUM(f.SalesAmount) AS INT) AS TotalVentas
FROM
    FactInternetSales AS f
WHERE 
    F.OrderDate > '2012-12-31' AND f.OrderDate < '2014-01-01'
GROUP BY
    CAST(f.OrderDate AS DATE)
ORDER BY
    CAST(f.OrderDate AS DATE) ASC;