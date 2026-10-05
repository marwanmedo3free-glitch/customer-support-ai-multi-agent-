DROP TABLE IF EXISTS tickets, refunds, orders, customers CASCADE;

CREATE TABLE customers (
  id SERIAL PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL);

CREATE TABLE orders (
  id INT PRIMARY KEY,
  customer_id INT NOT NULL REFERENCES customers(id),
  status TEXT NOT NULL CHECK (status IN ('processing','shipped','delivered','cancelled')),
  total NUMERIC(10,2) NOT NULL,
  address TEXT NOT NULL,
  tracking_eta DATE,
  delivered_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT now());

CREATE TABLE refunds (
  id SERIAL PRIMARY KEY,
  order_id INT NOT NULL REFERENCES orders(id),
  customer_id INT NOT NULL,
  amount NUMERIC(10,2) NOT NULL,
  reason TEXT,
  status TEXT NOT NULL DEFAULT 'issued',
  idempotency_key TEXT UNIQUE,
  created_at TIMESTAMPTZ DEFAULT now());

CREATE TABLE tickets (
  id SERIAL PRIMARY KEY, customer_id INT NOT NULL, summary TEXT,
  status TEXT NOT NULL DEFAULT 'open', created_at TIMESTAMPTZ DEFAULT now());

INSERT INTO customers (name, email) VALUES ('Alice','alice@example.com'), ('Bob','bob@example.com');

INSERT INTO orders (id, customer_id, status, total, address, tracking_eta, delivered_at) VALUES
 (1001, 1, 'shipped',    59.90, '1 Corniche St, Giza',  current_date + 2, NULL),
 (1002, 1, 'delivered',  89.00, '1 Corniche St, Giza',  NULL, now() - interval '5 days'),
 (1003, 1, 'delivered', 450.00, '1 Corniche St, Giza',  NULL, now() - interval '10 days'),
 (1004, 1, 'processing', 39.00, '1 Corniche St, Giza',  current_date + 5, NULL),
 (1005, 2, 'delivered',  40.00, '9 Hassan St, Cairo',   NULL, now() - interval '45 days'),
 (1006, 2, 'shipped',    75.00, '9 Hassan St, Cairo',   current_date + 1, NULL),
 (1007, 1, 'delivered', 600.00, '1 Corniche St, Giza',  NULL, now() - interval '3 days');
