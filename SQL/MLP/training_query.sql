SELECT 
    CAST(f.OrderDate AS DATE) AS sales_date,
    CAST(SUM(f.OrderQuantity) AS INT) AS TotalVentas
FROM 
    FactInternetSales AS f
WHERE 
    f.OrderDate < '2013-12-01'
    AND f.ProductKey = 214  
GROUP BY 
    CAST(f.OrderDate AS DATE)
ORDER BY 
    CAST(f.OrderDate AS DATE) ASC;